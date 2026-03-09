"""
ablation.py — Ablation study for rl-portfolio-agent.

Trains three degraded PPO variants on the validation split to quantify the
contribution of each design choice in the full model:

  1. no_tx_cost      — tests whether transaction cost penalisation matters.
                       Expected: the agent over-trades, hurting net returns
                       in practice even though in-sample reward looks higher.

  2. no_vol_features — tests whether rolling volatility signals add information.
                       Expected: the agent loses regime-awareness, leading to
                       worse risk-adjusted performance.

  3. raw_return      — tests whether the differential Sharpe reward (vs plain
                       portfolio return) improves risk-adjusted optimisation.
                       Expected: without Sharpe shaping the agent maximises
                       raw return, taking on more volatility for modest gain.

After training, all variants plus the saved full model are evaluated on the
validation split using the same run_validation logic as train.py.

Usage:
    python ablation.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, override

import numpy as np
import numpy.typing as npt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from baselines import evaluate_portfolio
from data import load_data
from environment import PortfolioEnv

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Shorter training budget than full training (500k) to keep ablation tractable
ABLATION_TIMESTEPS: int = 300_000

# Fixed hyperparameters — same values used as the default grid point in train.py
LEARNING_RATE: float = 3e-4
N_STEPS: int = 2048
ENT_COEF: float = 0.01

# Feature column indices of the volatility features in the (T, 15) array.
# data.py lays out columns as:
#   [asset0_logret, asset0_vol, asset0_meanrev,
#    asset1_logret, asset1_vol, asset1_meanrev, ...]
# The *_vol column for asset i sits at index i*3 + 1.
VOL_FEATURE_INDICES: list[int] = [1, 4, 7, 10, 13]

# Path where train.py saves the best model
BEST_MODEL_PATH: Path = Path("best_model/best_model")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

type FloatArray = npt.NDArray[np.float32]
type Metrics = dict[str, float]

# ---------------------------------------------------------------------------
# Ablation variant 1 helper — zeroed volatility features
# ---------------------------------------------------------------------------


def _zero_vol_features(features: FloatArray) -> FloatArray:
    """
    Return a copy of the feature array with all volatility columns zeroed out.

    This removes the rolling-volatility signal from the observation without
    altering the array dimensions, so the environment and model architecture
    remain identical. The agent still receives logret and meanrev signals.

    Parameters
    ----------
    features:
        Normalised feature array of shape (T, 15).

    Returns
    -------
    Feature array of the same shape with VOL_FEATURE_INDICES columns set to 0.
    """
    zeroed = features.copy()
    zeroed[:, VOL_FEATURE_INDICES] = 0.0
    return zeroed


# ---------------------------------------------------------------------------
# Ablation variant 3 — raw return reward environment
# ---------------------------------------------------------------------------


class RawReturnEnv(PortfolioEnv):
    """
    PortfolioEnv variant that replaces the differential Sharpe reward with a
    plain portfolio-return reward net of transaction costs:

        reward = portfolio_return - transaction_cost

    All other logic (observation, action space, termination) is unchanged.
    This isolates the contribution of the differential Sharpe shaping: without
    it the agent is free to take on as much variance as it likes, since only
    mean return enters the reward signal.
    """

    @override
    def step(
        self, action: FloatArray
    ) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        """
        Override step() to swap the differential Sharpe reward for a raw
        portfolio-return reward.  All bookkeeping (weight update, timestep
        advance, obs construction) is inherited from PortfolioEnv.step() via
        super(), but we replace the reward before returning.
        """
        # Call the parent step to get all state updates and the info dict.
        # The parent reward (differential Sharpe - cost) will be discarded.
        obs, _parent_reward, terminated, truncated, info = super().step(action)

        # Reconstruct reward as raw portfolio return minus transaction cost.
        # Both values are already computed by the parent and stored in info.
        portfolio_return: float = float(info["portfolio_return"])
        transaction_cost: float = float(info["transaction_cost"])
        reward: float = portfolio_return - transaction_cost

        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Validation helper (copied from train.py — not imported to avoid circular deps)
# ---------------------------------------------------------------------------


def run_validation(
    model: PPO,
    val_features: FloatArray,
    val_prices: FloatArray,
    env_cls: type[PortfolioEnv] = PortfolioEnv,
    env_kwargs: dict[str, Any] | None = None,
) -> Metrics:
    """
    Run one deterministic episode on the validation environment and return
    a full metrics dict (not just Sharpe, unlike the train.py version).

    The episode starts at the earliest valid timestep (index = window) so it
    covers the entire validation period consistently across all variants.

    Parameters
    ----------
    model:
        Trained PPO model.
    val_features:
        Normalised feature array for the validation split, shape (T_val, 15).
    val_prices:
        Raw prices for the validation split, shape (T_val, 5).
    env_cls:
        Environment class to instantiate (allows RawReturnEnv to be used here).
    env_kwargs:
        Extra keyword arguments forwarded to the environment constructor.

    Returns
    -------
    Dict from evaluate_portfolio with annualised_return, annualised_volatility,
    sharpe_ratio, max_drawdown, and calmar_ratio.
    """
    kwargs: dict[str, Any] = env_kwargs or {}
    env = env_cls(val_features, val_prices, **kwargs)

    obs, _ = env.reset(seed=0)
    env._t = env._window  # type: ignore[attr-defined]  # bypass random start
    obs = env._get_obs()  # type: ignore[attr-defined]  # rebuild obs at new _t

    _T_val, N = val_prices.shape
    current_t: int = env._window  # type: ignore[attr-defined]

    step_prices: list[FloatArray] = [val_prices[current_t]]
    weights_list: list[FloatArray] = [
        np.full(N, 1.0 / N, dtype=np.float32)  # dummy anchor; unused in return calc
    ]

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        current_t += 1
        weights_list.append(info["weights"])
        step_prices.append(val_prices[current_t])

    if len(weights_list) < 2:
        logger.warning("Validation episode was too short to compute metrics.")
        return {
            "annualised_return": 0.0,
            "annualised_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
        }

    weights_arr: npt.NDArray[np.float64] = np.stack(weights_list).astype(np.float64)
    prices_arr: npt.NDArray[np.float64] = np.stack(step_prices).astype(np.float64)

    return evaluate_portfolio(prices_arr, weights_arr)


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------


def _train_variant(
    name: str,
    train_features: FloatArray,
    train_prices: FloatArray,
    env_cls: type[PortfolioEnv] = PortfolioEnv,
    env_kwargs: dict[str, Any] | None = None,
) -> PPO:
    """
    Train a PPO agent for one ablation variant.

    Parameters
    ----------
    name:
        Human-readable variant name, used only for logging.
    train_features:
        Feature array to train on (may be the zeroed-vol version).
    train_prices:
        Raw price array for the training split.
    env_cls:
        Environment class to use (PortfolioEnv or RawReturnEnv).
    env_kwargs:
        Extra keyword arguments forwarded to the environment constructor.

    Returns
    -------
    Trained PPO model.
    """
    logger.info(f"Training ablation variant: {name}")

    kwargs: dict[str, Any] = env_kwargs or {}
    train_env: Monitor = Monitor(
        env_cls(train_features, train_prices, **kwargs),
        filename=None,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        ent_coef=ENT_COEF,
        verbose=0,
    )

    model.learn(total_timesteps=ABLATION_TIMESTEPS, reset_num_timesteps=True)
    logger.info(f"Finished training: {name}")

    return model


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------


def _print_table(rows: list[tuple[str, Metrics]]) -> None:
    """
    Print a formatted comparison table for all ablation variants.

    Parameters
    ----------
    rows:
        List of (variant_name, metrics_dict) pairs in display order.
    """
    col_name = max(len(r[0]) for r in rows)
    col_name = max(col_name, len("Variant"))

    header = (
        f"{'Variant':<{col_name}}  "
        f"{'Ann. Return':>12}  "
        f"{'Ann. Vol':>10}  "
        f"{'Sharpe':>8}  "
        f"{'Max DD':>10}"
    )
    divider = "-" * len(header)

    print(f"\n{divider}")
    print(header)
    print(divider)

    for name, m in rows:
        print(
            f"{name:<{col_name}}  "
            f"{m['annualised_return']:>11.2%}  "
            f"{m['annualised_volatility']:>9.2%}  "
            f"{m['sharpe_ratio']:>8.3f}  "
            f"{m['max_drawdown']:>9.2%}"
        )

    print(divider)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Run all three ablation variants, optionally load the full model, evaluate
    each on the validation set, and print a comparison table.
    """
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading data ...")
    data = load_data()

    train_features: FloatArray = data["train"]
    train_prices: FloatArray = data["train_prices"]
    val_features: FloatArray = data["val"]
    val_prices: FloatArray = data["val_prices"]

    # ------------------------------------------------------------------
    # 2. Prepare zeroed-vol features (used by both train and val for the
    #    no_vol_features variant so the agent never sees volatility signal).
    # ------------------------------------------------------------------
    train_features_no_vol = _zero_vol_features(train_features)
    val_features_no_vol = _zero_vol_features(val_features)

    # ------------------------------------------------------------------
    # 3. Train all three ablation variants
    # ------------------------------------------------------------------

    # Variant 1: no transaction cost.
    # Tests whether penalising turnover is important. Without it the reward
    # no longer discourages churning, so we expect higher turnover and lower
    # net returns when accounting for real costs.
    model_no_tx = _train_variant(
        name="no_tx_cost",
        train_features=train_features,
        train_prices=train_prices,
        env_cls=PortfolioEnv,
        env_kwargs={"transaction_cost": 0.0},
    )

    # Variant 2: no volatility features.
    # Tests whether the rolling-vol columns contribute meaningful signal.
    # The zeroed features are used for both training and evaluation so the
    # agent is evaluated in the same conditions it was trained under.
    model_no_vol = _train_variant(
        name="no_vol_features",
        train_features=train_features_no_vol,
        train_prices=train_prices,
        env_cls=PortfolioEnv,
    )

    # Variant 3: raw return reward.
    # Tests whether differential Sharpe shaping improves risk-adjusted outcomes.
    # The agent optimises mean return directly, so we expect it to take on more
    # volatility without a proportionate gain in return.
    model_raw = _train_variant(
        name="raw_return",
        train_features=train_features,
        train_prices=train_prices,
        env_cls=RawReturnEnv,
    )

    # ------------------------------------------------------------------
    # 4. Evaluate all variants on the validation split
    # ------------------------------------------------------------------
    logger.info("Evaluating all variants on validation set ...")

    rows: list[tuple[str, Metrics]] = []

    metrics_no_tx = run_validation(
        model_no_tx,
        val_features,
        val_prices,
        env_cls=PortfolioEnv,
        env_kwargs={"transaction_cost": 0.0},
    )
    rows.append(("no_tx_cost", metrics_no_tx))

    # no_vol_features: evaluate with zeroed vol features to match training conditions
    metrics_no_vol = run_validation(
        model_no_vol,
        val_features_no_vol,
        val_prices,
        env_cls=PortfolioEnv,
    )
    rows.append(("no_vol_features", metrics_no_vol))

    metrics_raw = run_validation(
        model_raw,
        val_features,
        val_prices,
        env_cls=RawReturnEnv,
    )
    rows.append(("raw_return", metrics_raw))

    # ------------------------------------------------------------------
    # 5. Optionally load and evaluate the full best model for comparison
    # ------------------------------------------------------------------
    best_model_zip = Path(f"{BEST_MODEL_PATH}.zip")
    if best_model_zip.exists():
        logger.info(f"Loading full model from {BEST_MODEL_PATH} ...")
        with warnings.catch_warnings():
            # SB3 may emit a UserWarning about policy kwargs on load; suppress it
            warnings.simplefilter("ignore", UserWarning)
            full_model = PPO.load(str(BEST_MODEL_PATH))

        metrics_full = run_validation(full_model, val_features, val_prices)
        # Prepend so the full model row appears first in the table
        rows.insert(0, ("full_model", metrics_full))
    else:
        logger.warning(
            f"Best model not found at {best_model_zip} — skipping full_model row. "
            "Run train.py first to generate it."
        )

    # ------------------------------------------------------------------
    # 6. Print comparison table
    # ------------------------------------------------------------------
    _print_table(rows)


if __name__ == "__main__":
    main()
