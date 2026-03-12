"""
train.py — PPO training with TensorBoard logging and checkpointing.

Trains a single PPO agent on the training split, evaluates on the validation
split, and saves the best model checkpoint.

Usage:
    python train.py
    # or via the project venv:
    .venv/bin/python train.py
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, override

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from clearml import Task
from pydantic import BaseModel, ConfigDict, Field
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import pandas as pd

from baselines import evaluate_portfolio
from data import TICKERS, load_data
from environment import PortfolioEnv

# ---------------------------------------------------------------------------
# Module-level constants — change here to retrain with different settings
# ---------------------------------------------------------------------------

# Total environment steps for training
TOTAL_TIMESTEPS: int = 1_500_000

# Number of parallel environments for data collection.
# Each env steps independently from a different random start, giving
# n_envs * n_steps decorrelated transitions per PPO update.
N_ENVS: int = 8

# Optional path to a saved model to warm-start from (e.g. "best_model/best_model").
# If set, loads weights + optimizer state instead of initialising from scratch.
# Set to None to train from random initialisation.
WARM_START_PATH: str | None = "/app/runs/warm_start/best_model"

# How often (in steps) to save a model checkpoint
CHECKPOINT_FREQ: int = 50_000

# Hyperparameters — chosen by the user based on validation results
LEARNING_RATE: float = 1e-4
N_STEPS: int = 2048
ENT_COEF: float = 0.01

# Proportional transaction cost applied to L1 weight change at each rebalance step.
# 0.001 = 10 bps, a realistic proxy for ETF bid-ask spreads.
TRANSACTION_COST: float = 0.001

# Annualisation factor for Sharpe computation inside the callback
TRADING_DAYS_PER_YEAR: int = 252

# Root output directory — always resolves to <project_root>/runs/ regardless of cwd
_RUNS_DIR: Path = Path(__file__).resolve().parent.parent / "runs"
LOG_DIR: Path = _RUNS_DIR / "logs"
CHECKPOINT_DIR: Path = _RUNS_DIR / "checkpoints"
BEST_MODEL_DIR: Path = _RUNS_DIR / "best_model"

# ClearML project name — must match the project created by ml-setup
CLEARML_PROJECT: str = "rl-portfolio-agent"

_VAL_EVENTS: list[tuple[str, str]] = [
    ("2015-China-selloff", "2015-08-24"),
    ("2016-US-election", "2016-11-08"),
    ("2018-Q4-crash", "2018-10-01"),
]

_TRAIN_EVENTS: list[tuple[str, str]] = [
    ("Train-sample-A", "2003-06-01"),
    ("Train-sample-B", "2008-09-01"),
    ("Train-sample-C", "2012-01-01"),
]

_VAL_REGIMES: dict[str, tuple[str, str]] = {
    "Bull 2015–2016": ("2015-02-01", "2015-07-31"),
    "China selloff": ("2015-08-01", "2016-01-31"),
    "Trump rally": ("2016-11-09", "2017-12-31"),
    "Q4 2018 crash": ("2018-10-01", "2018-12-31"),
}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _make_env(features: FloatArray, prices: FloatArray, rank: int) -> Monitor:
    """Factory for SubprocVecEnv — must be module-level to be picklable."""
    env = PortfolioEnv(features, prices, transaction_cost=TRANSACTION_COST)
    env.reset(seed=rank)
    return Monitor(env)


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

# ---------------------------------------------------------------------------
# Config model — one instance per hyperparameter combination
# ---------------------------------------------------------------------------


class RunConfig(BaseModel):
    """Immutable config for a single training run."""

    model_config = ConfigDict(frozen=True)

    learning_rate: float = Field(gt=0, description="Adam learning rate")
    n_steps: int = Field(ge=1, description="PPO rollout buffer length in steps")
    ent_coef: float = Field(ge=0, description="Entropy regularisation coefficient")

    @property
    def run_name(self) -> str:
        """Canonical name used for checkpoint and TensorBoard directories."""
        return f"lr{self.learning_rate}_steps{self.n_steps}_ent{self.ent_coef}"


# ---------------------------------------------------------------------------
# Transaction cost curriculum callback
# ---------------------------------------------------------------------------


class TxCostCurriculumCallback(BaseCallback):
    """
    Ramps transaction_cost on all training envs from _TC_MIN to TRANSACTION_COST
    over the full run using a quadratic schedule. Lets the policy learn basic
    allocation before being penalised for excessive turnover.
    """

    _TC_MIN: float = 0.0002

    def __init__(self, total_timesteps: int, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._total_timesteps = total_timesteps

    @override
    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self._total_timesteps, 1.0)
        tc = self._TC_MIN + (TRANSACTION_COST - self._TC_MIN) * (progress**2)
        for monitor in self.training_env.envs:  # type: ignore[attr-defined]
            monitor.env.transaction_cost = tc  # type: ignore[attr-defined]
        return True


# ---------------------------------------------------------------------------
# Custom training callback
# ---------------------------------------------------------------------------


class TrainingCallback(BaseCallback):
    """
    Per-episode metric logger for ClearML and TensorBoard.

    At every step collects portfolio_return, transaction_cost, and weights
    from info. At episode end logs:
      - Sharpe, turnover, reward stats (mean/std/min/max)
      - Weight entropy (policy concentration)
      - Per-asset mean allocation
      - Gross return vs transaction cost drag
    """

    # Small epsilon to avoid log(0) in entropy calculation
    _ENTROPY_EPS: float = 1e-8

    def __init__(self, task: Task, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._task = task
        self._episode_returns: list[float] = []
        self._episode_costs: list[float] = []
        self._episode_weights: list[FloatArray] = []
        self._episode_weight_changes: list[float] = []
        self._prev_weights: FloatArray | None = None
        self._episode_count: int = 0

    @override
    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [{}])
        dones: list[bool] = self.locals.get("dones", [False])

        for i, info in enumerate(infos):
            ret: float = float(info.get("portfolio_return", 0.0))
            cost: float = float(info.get("transaction_cost", 0.0))
            weights: FloatArray = info.get(
                "weights", np.full(PortfolioEnv.N_ASSETS, 1.0 / PortfolioEnv.N_ASSETS)
            )

            self._episode_returns.append(ret)
            self._episode_costs.append(cost)
            self._episode_weights.append(weights.copy())

            if self._prev_weights is not None:
                self._episode_weight_changes.append(
                    float(np.sum(np.abs(weights - self._prev_weights)))
                )
            self._prev_weights = weights.copy()

            if dones[i]:
                self._log_episode_metrics()
                self._reset_episode()

        return True

    def _log_episode_metrics(self) -> None:
        if len(self._episode_returns) < 2:
            return

        n = self.num_timesteps  # x-axis for ClearML scalars
        cl = self._task.get_logger()

        returns = np.array(self._episode_returns, dtype=np.float64)
        weights_arr = np.stack(self._episode_weights)  # (T, N)

        # --- Sharpe ---
        std = float(returns.std(ddof=1))
        sharpe = (
            float(returns.mean()) / std * math.sqrt(TRADING_DAYS_PER_YEAR)
            if std > 0.0
            else 0.0
        )

        # --- Reward distribution ---
        reward_mean = float(returns.mean())
        reward_std = float(returns.std(ddof=1))

        # --- Turnover ---
        turnover = (
            float(np.mean(self._episode_weight_changes))
            if self._episode_weight_changes
            else 0.0
        )

        # --- Weight entropy: -sum(w * log(w)), averaged over episode steps ---
        # High entropy ≈ equal weight (not learning to differentiate assets)
        # Low entropy ≈ concentrated (potentially overfit to one asset)
        entropy_per_step = -np.sum(
            weights_arr * np.log(weights_arr + self._ENTROPY_EPS), axis=1
        )
        mean_entropy = float(entropy_per_step.mean())

        # --- Per-asset mean allocation ---
        mean_weights = weights_arr.mean(axis=0)  # shape (N,)

        # --- Transaction cost drag vs gross return ---
        mean_cost = float(np.mean(self._episode_costs))

        # Log directly to ClearML only — not via self.logger.record() to avoid
        # polluting SB3's internal "train/" TensorBoard group with our metrics.
        for title, series, value in [
            ("reward", "mean", reward_mean),
            ("reward", "std", reward_std),
            ("policy", "sharpe", sharpe),
            ("policy", "turnover", turnover),
            ("policy", "weight_entropy", mean_entropy),
            ("costs", "mean_tx_cost_per_step", mean_cost),
            ("costs", "gross_return_per_step", reward_mean),
        ]:
            cl.report_scalar(title=title, series=series, value=value, iteration=n)

        # Per-asset mean allocation as individual series
        for ticker, w in zip(TICKERS, mean_weights):
            cl.report_scalar(
                title="asset_allocation", series=ticker, value=float(w), iteration=n
            )

        self._episode_count += 1

    def _reset_episode(self) -> None:
        self._episode_returns = []
        self._episode_costs = []
        self._episode_weights = []
        self._episode_weight_changes = []
        self._prev_weights = None


# ---------------------------------------------------------------------------
# Periodic validation callback — logs val Sharpe at each checkpoint
# ---------------------------------------------------------------------------


class PeriodicValCallback(BaseCallback):
    """
    Runs a deterministic validation episode every `eval_freq` steps and logs
    val Sharpe to ClearML. This creates a train-vs-val Sharpe curve so you
    can see when overfitting begins.
    """

    def __init__(
        self,
        val_features: FloatArray,
        val_prices: FloatArray,
        task: Task,
        best_model_path: str,
        eval_freq: int = CHECKPOINT_FREQ,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._val_features = val_features
        self._val_prices = val_prices
        self._task = task
        self._best_model_path = best_model_path
        self._eval_freq = eval_freq
        self._best_sharpe: float = -float("inf")

    @override
    def _on_step(self) -> bool:
        if self.num_timesteps % self._eval_freq == 0:
            val_sharpe = _run_validation_sharpe(
                self.model, self._val_features, self._val_prices  # type: ignore[arg-type]
            )
            self._task.get_logger().report_scalar(
                title="validation",
                series="sharpe_ratio",
                value=val_sharpe,
                iteration=self.num_timesteps,
            )
            if val_sharpe > self._best_sharpe:
                self._best_sharpe = val_sharpe
                self.model.save(self._best_model_path)  # type: ignore[union-attr]
                logger.info(
                    f"  [step {self.num_timesteps:>7}]  val Sharpe = {val_sharpe:.4f}  *** new best — checkpoint saved"
                )
            else:
                logger.info(
                    f"  [step {self.num_timesteps:>7}]  val Sharpe = {val_sharpe:.4f}"
                )
        return True


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _rollout(
    model: PPO,
    features: FloatArray,
    prices: FloatArray,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Run a deterministic episode from the earliest valid timestep.

    Returns (weights_arr, prices_arr) aligned for evaluate_portfolio:
    weights_arr[i] earns log(prices_arr[i] / prices_arr[i-1]), row 0 is a
    dummy anchor.
    """
    _, N = prices.shape
    env = PortfolioEnv(features, prices, transaction_cost=TRANSACTION_COST)
    obs, _ = env.reset(seed=0)
    env._t = env._window  # type: ignore[attr-defined]
    obs = env._get_obs()  # type: ignore[attr-defined]

    current_t: int = env._window  # type: ignore[attr-defined]
    step_prices: list[FloatArray] = [prices[current_t]]
    weights_list: list[FloatArray] = [np.full(N, 1.0 / N, dtype=np.float32)]

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        current_t += 1
        weights_list.append(info["weights"])
        step_prices.append(prices[current_t])

    weights_arr = np.stack(weights_list).astype(np.float64)
    prices_arr = np.stack(step_prices).astype(np.float64)
    return weights_arr, prices_arr


def _run_validation_sharpe(
    model: PPO,
    val_features: FloatArray,
    val_prices: FloatArray,
) -> float:
    """Lightweight helper used by PeriodicValCallback during training."""
    weights_arr, prices_arr = _rollout(model, val_features, val_prices)
    if len(weights_arr) < 2:
        return 0.0
    return float(evaluate_portfolio(prices_arr, weights_arr)["sharpe_ratio"])


def _plot_event_zoom(
    weights_arr: npt.NDArray[np.float64],
    prices_arr: npt.NDArray[np.float64],
    episode_dates: pd.DatetimeIndex,
    tickers: list[str],
    task: Task,
    final_step: int,
    events: list[tuple[str, str]] = _VAL_EVENTS,
    title: str = "event_zoom",
) -> None:
    cl = task.get_logger()
    cmap = plt.colormaps["tab10"]
    colors = [cmap(i) for i in range(len(tickers))]

    log_rets = np.log(prices_arr[1:] / prices_arr[:-1])
    port_rets = np.sum(weights_arr[1:] * log_rets, axis=1)
    ew_rets = np.sum(
        np.full_like(weights_arr[1:], 1.0 / len(tickers)) * log_rets, axis=1
    )

    # Strip anchor row — weights_arr[0] is equal-weight placeholder, not a real decision.
    # After stripping, weights_real[i] aligns with episode_dates[i] and port_rets[i]
    # is the return earned from episode_dates[i] to episode_dates[i+1].
    weights_real = weights_arr[1:]
    dates_real = episode_dates[1:]

    for event_label, center_date in events:
        center_ts = pd.Timestamp(center_date)
        if center_ts < dates_real[0] or center_ts > dates_real[-1]:
            continue

        center_idx = int(dates_real.searchsorted(center_ts))
        lo = max(0, center_idx - 30)
        hi = min(len(dates_real) - 1, center_idx + 30)

        w_window = weights_real[lo : hi + 1]
        dates_window = dates_real[lo : hi + 1]

        # port_rets[lo:hi] covers the returns from dates_real[lo] to dates_real[hi],
        # giving hi-lo values. Prepending 1.0 anchors cumulative value at dates_real[lo].
        port_window = port_rets[lo:hi]
        ew_window = ew_rets[lo:hi]

        agent_cum = np.concatenate([[1.0], np.exp(np.cumsum(port_window))])
        ew_cum = np.concatenate([[1.0], np.exp(np.cumsum(ew_window))])

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6))

        ax_top.stackplot(
            dates_window,
            w_window.T,
            labels=tickers,
            colors=colors,
            alpha=0.85,
        )
        ax_top.set_ylim(0, 1)
        ax_top.set_ylabel("Weight")
        ax_top.set_title(f"Event zoom: {event_label}")
        ax_top.legend(loc="upper right", fontsize=7, ncol=len(tickers))

        ax_bot.plot(dates_window, agent_cum, label="Agent")
        ax_bot.plot(dates_window, ew_cum, label="Equal weight", linestyle="--")
        ax_bot.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
        ax_bot.set_ylabel("Cumulative value")
        ax_bot.legend(fontsize=8)

        fig.tight_layout()
        cl.report_matplotlib_figure(
            title=title,
            series=event_label,
            figure=fig,
            iteration=final_step,
        )
        plt.close(fig)


def _plot_regime_allocations(
    weights_arr: npt.NDArray[np.float64],
    episode_dates: pd.DatetimeIndex,
    tickers: list[str],
    task: Task,
    final_step: int,
) -> None:
    cl = task.get_logger()

    # Strip anchor row — weights_arr[0] is equal-weight placeholder, not a real decision.
    weights_real = weights_arr[1:]
    dates_real = episode_dates[1:]

    regime_means: dict[str, npt.NDArray[np.float64]] = {}
    for regime_label, (start, end) in _VAL_REGIMES.items():
        mask = (dates_real >= pd.Timestamp(start)) & (dates_real <= pd.Timestamp(end))
        if not mask.any():
            continue
        sliced = weights_real[mask]
        regime_means[regime_label] = sliced.mean(axis=0)

    if not regime_means:
        return

    n_regimes = len(regime_means)
    n_assets = len(tickers)
    x = np.arange(n_regimes)
    width = 0.8 / n_assets

    fig, ax = plt.subplots(figsize=(max(8, n_regimes * 2), 5))
    for i, ticker in enumerate(tickers):
        vals = [regime_means[r][i] for r in regime_means]
        ax.bar(x + i * width, vals, width, label=ticker)

    ax.set_xticks(x + width * (n_assets - 1) / 2)
    ax.set_xticklabels(list(regime_means.keys()), rotation=15, ha="right")
    ax.set_ylabel("Mean weight")
    ax.set_ylim(0, 1)
    ax.set_title("Mean allocation by market regime")
    ax.legend(fontsize=8, ncol=n_assets)
    fig.tight_layout()
    cl.report_matplotlib_figure(
        title="validation_plots",
        series="regime_allocations",
        figure=fig,
        iteration=final_step,
    )
    plt.close(fig)


def _plot_turnover_timeline(
    weights_arr: npt.NDArray[np.float64],
    prices_arr: npt.NDArray[np.float64],
    episode_dates: pd.DatetimeIndex,
    task: Task,
    final_step: int,
) -> None:
    cl = task.get_logger()

    turnover = np.abs(np.diff(weights_arr, axis=0)).sum(axis=1)  # shape (T-1,)

    log_rets = np.log(prices_arr[1:] / prices_arr[:-1])
    port_rets = np.sum(weights_arr[1:] * log_rets, axis=1)
    cum_value = np.exp(np.cumsum(port_rets))

    # turnover and cum_value are length T-1; align with episode_dates[1:]
    dates_ret = episode_dates[1 : 1 + len(cum_value)]
    if len(dates_ret) == 0:
        return

    fig, ax_left = plt.subplots(figsize=(14, 4))
    ax_right = ax_left.twinx()

    ax_left.plot(
        dates_ret, cum_value, color="steelblue", linewidth=1.5, label="Portfolio value"
    )
    ax_left.set_ylabel("Cumulative value")

    ax_right.fill_between(
        dates_ret, turnover, alpha=0.3, color="sandybrown", label="Turnover"
    )
    ax_right.set_ylabel("Daily turnover (L1)")

    for event_label, center_date in _VAL_EVENTS:
        center_ts = pd.Timestamp(center_date)
        if dates_ret[0] <= center_ts <= dates_ret[-1]:
            ax_left.axvline(center_ts, color="grey", linestyle=":", linewidth=0.9)
            ax_left.text(
                center_ts,
                float(cum_value.max()),
                event_label,
                rotation=90,
                va="top",
                fontsize=7,
                color="grey",
            )

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, fontsize=8)
    ax_left.set_title("Cumulative portfolio value and daily turnover")
    fig.tight_layout()
    cl.report_matplotlib_figure(
        title="validation_plots",
        series="turnover_timeline",
        figure=fig,
        iteration=final_step,
    )
    plt.close(fig)


def run_validation(
    model: PPO,
    val_features: FloatArray,
    val_prices: FloatArray,
    val_dates: pd.DatetimeIndex,
    task: Task,
) -> dict[str, float]:
    """
    Full post-training validation: runs the deterministic policy on the val
    split, logs all metrics and plots to ClearML, and returns the metrics dict.
    """
    weights_arr, prices_arr = _rollout(model, val_features, val_prices)

    if len(weights_arr) < 2:
        logger.warning("Validation episode was too short.")
        return {}

    window = 20
    episode_dates: pd.DatetimeIndex = val_dates[window : window + len(weights_arr)]

    metrics = evaluate_portfolio(prices_arr, weights_arr)
    cl = task.get_logger()

    # --- Scalar metrics ---
    final_step = len(weights_arr)
    for key, value in metrics.items():
        if value != float("inf"):
            cl.report_scalar(
                title="validation_final", series=key, value=value, iteration=final_step
            )

    # --- Weight allocation heatmap (assets × time) ---
    fig, ax = plt.subplots(figsize=(14, 3))
    im = ax.imshow(
        weights_arr[1:].T,  # skip anchor row; shape (N, T)
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
    )
    ax.set_yticks(range(len(TICKERS)))
    ax.set_yticklabels(TICKERS)
    ax.set_xlabel("Validation step")
    ax.set_title("Agent weight allocation over validation period")
    fig.colorbar(im, ax=ax, label="Weight")
    fig.tight_layout()
    cl.report_matplotlib_figure(
        title="validation_plots",
        series="weight_heatmap",
        figure=fig,
        iteration=final_step,
    )
    plt.close(fig)

    # --- Per-asset allocation histogram ---
    fig, ax = plt.subplots(figsize=(7, 4))
    mean_w = weights_arr[1:].mean(axis=0)
    ax.bar(TICKERS, mean_w)
    ax.set_ylabel("Mean weight")
    ax.set_title("Mean portfolio allocation over validation period")
    ax.set_ylim(0, 1)
    for i, v in enumerate(mean_w):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    cl.report_matplotlib_figure(
        title="validation_plots",
        series="mean_allocation",
        figure=fig,
        iteration=final_step,
    )
    plt.close(fig)

    # --- Cumulative portfolio value ---
    log_rets = np.log(prices_arr[1:] / prices_arr[:-1])
    port_rets = np.sum(weights_arr[1:] * log_rets, axis=1)
    cum_value = np.exp(np.cumsum(port_rets))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(cum_value)
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Validation step")
    ax.set_ylabel("Portfolio value (normalised to 1.0)")
    ax.set_title("Cumulative portfolio value — validation period")
    fig.tight_layout()
    cl.report_matplotlib_figure(
        title="validation_plots",
        series="cumulative_value",
        figure=fig,
        iteration=final_step,
    )
    plt.close(fig)

    # --- Event-zoom plots ---
    _plot_event_zoom(weights_arr, prices_arr, episode_dates, TICKERS, task, final_step)

    # --- Regime allocation bar chart ---
    _plot_regime_allocations(weights_arr, episode_dates, TICKERS, task, final_step)

    # --- Turnover + cumulative value dual-axis ---
    _plot_turnover_timeline(weights_arr, prices_arr, episode_dates, task, final_step)

    return metrics


# ---------------------------------------------------------------------------
# Training loop for a single hyperparameter combination
# ---------------------------------------------------------------------------


def _train_one(
    cfg: RunConfig,
    train_features: FloatArray,
    train_prices: FloatArray,
    train_dates: pd.DatetimeIndex,
    val_features: FloatArray,
    val_prices: FloatArray,
    val_dates: pd.DatetimeIndex,
) -> tuple[PPO, float]:
    """
    Train a PPO agent for one hyperparameter combination.

    Creates the environment, wraps it with Monitor, creates the PPO model,
    attaches checkpointing and custom callbacks, trains, then evaluates on
    the validation set.

    Parameters
    ----------
    cfg:
        Immutable run configuration.
    train_features, train_prices:
        Training split arrays.
    val_features, val_prices:
        Validation split arrays.

    Returns
    -------
    Trained PPO model and its validation Sharpe ratio.
    """
    logger.info(f"Starting run: {cfg.run_name}")

    # ------------------------------------------------------------------
    # 0. Initialise ClearML task
    #
    # Task.init() must be called before model.learn() so that ClearML can
    # intercept the TensorBoard logger and mirror all SB3 metrics (episode
    # reward, episode_sharpe, episode_turnover) to the ClearML dashboard.
    # Each hyperparameter combination gets its own task so runs are
    # independently comparable in the ClearML UI.
    # ------------------------------------------------------------------
    task = Task.init(
        project_name=CLEARML_PROJECT,
        task_name=cfg.run_name,
        reuse_last_task_id=False,
    )
    # Log the hyperparameters so they appear in the ClearML HP panel
    task.connect(cfg.model_dump(), name="hyperparameters")

    # ------------------------------------------------------------------
    # 1. Build and wrap the training environment
    # ------------------------------------------------------------------
    train_env = DummyVecEnv(
        [
            lambda rank=i: _make_env(train_features, train_prices, rank)
            for i in range(N_ENVS)
        ]
    )

    # ------------------------------------------------------------------
    # 2. Create the PPO model
    # ------------------------------------------------------------------
    tensorboard_log_dir = str(LOG_DIR / cfg.run_name)

    if WARM_START_PATH is not None:
        model = PPO.load(
            WARM_START_PATH,
            env=train_env,
            tensorboard_log=tensorboard_log_dir,
        )
        model.learning_rate = cfg.learning_rate
        model.n_steps = cfg.n_steps
        model.ent_coef = cfg.ent_coef
        logger.info(f"Warm-starting from {WARM_START_PATH}")
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            ent_coef=cfg.ent_coef,
            tensorboard_log=tensorboard_log_dir,
            verbose=0,
        )

    # ------------------------------------------------------------------
    # 3. Build callbacks
    # ------------------------------------------------------------------
    checkpoint_path = CHECKPOINT_DIR / cfg.run_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(checkpoint_path),
        name_prefix="ppo_checkpoint",
        verbose=0,
    )
    training_cb = TrainingCallback(task=task, verbose=0)
    best_checkpoint_path = str(BEST_MODEL_DIR / "best_checkpoint")
    val_cb = PeriodicValCallback(
        val_features=val_features,
        val_prices=val_prices,
        task=task,
        best_model_path=best_checkpoint_path,
        eval_freq=CHECKPOINT_FREQ,
    )
    curriculum_cb = TxCostCurriculumCallback(total_timesteps=TOTAL_TIMESTEPS)

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, training_cb, val_cb, curriculum_cb],
        tb_log_name="run",
        reset_num_timesteps=True,
        progress_bar=True,
    )

    # ------------------------------------------------------------------
    # 5. Load best checkpoint if it improved on the final model
    # ------------------------------------------------------------------
    best_checkpoint_file = Path(best_checkpoint_path + ".zip")
    if best_checkpoint_file.exists() and val_cb._best_sharpe > -float("inf"):
        model = PPO.load(best_checkpoint_path)
        logger.info(
            f"  Loaded best checkpoint (val Sharpe {val_cb._best_sharpe:.4f}) for final evaluation"
        )

    # ------------------------------------------------------------------
    # 6. Full post-training validation — logs all metrics + plots
    # ------------------------------------------------------------------
    metrics = run_validation(model, val_features, val_prices, val_dates, task)
    val_sharpe = float(metrics.get("sharpe_ratio", 0.0))
    logger.info(f"  {cfg.run_name}  val Sharpe = {val_sharpe:.4f}")

    # Training event zoom — same plot as validation to compare in-sample vs out-of-sample behaviour
    train_weights_arr, train_prices_arr = _rollout(model, train_features, train_prices)
    window = 20
    train_episode_dates: pd.DatetimeIndex = train_dates[
        window : window + len(train_weights_arr)
    ]
    _plot_event_zoom(
        train_weights_arr,
        train_prices_arr,
        train_episode_dates,
        TICKERS,
        task,
        len(train_weights_arr),
        events=_TRAIN_EVENTS,
        title="train_event_zoom",
    )

    # Save and upload model artifact on the same task — no separate task needed
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = BEST_MODEL_DIR / "best_model"
    model.save(str(save_path))
    task.upload_artifact(
        name="best_model",
        artifact_object=save_path.with_suffix(".zip"),
    )

    task.close()
    return model, val_sharpe


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Train a single PPO agent, evaluate on validation, and save the model.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading data ...")
    data = load_data()

    train_features: FloatArray = data["train"]
    train_prices: FloatArray = data["train_prices"]
    train_dates: pd.DatetimeIndex = data["train_dates"]
    val_features: FloatArray = data["val"]
    val_prices: FloatArray = data["val_prices"]
    val_dates: pd.DatetimeIndex = data["val_dates"]

    # ------------------------------------------------------------------
    # 2. Build config from module-level constants
    # ------------------------------------------------------------------
    cfg = RunConfig.model_validate(
        {"learning_rate": LEARNING_RATE, "n_steps": N_STEPS, "ent_coef": ENT_COEF}
    )

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    model, val_sharpe = _train_one(
        cfg,
        train_features,
        train_prices,
        train_dates,
        val_features,
        val_prices,
        val_dates,
    )

    logger.info(
        f"Model saved to {BEST_MODEL_DIR / 'best_model'}  (val_sharpe={val_sharpe:.4f})"
    )
    print(f"\nVal Sharpe: {val_sharpe:.4f}")
    print(f"Model saved to {BEST_MODEL_DIR / 'best_model'}\n")


if __name__ == "__main__":
    main()
