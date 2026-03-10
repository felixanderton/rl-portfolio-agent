"""
evaluate.py — Hold-out test evaluation for rl-portfolio-agent.

Loads the best PPO checkpoint, runs a single deterministic rollout on the
held-out test split, computes performance metrics for the agent and three
baselines, prints a formatted table, bootstraps a Sharpe CI, and saves three
diagnostic plots.

Usage:
    python evaluate.py
    # or:
    .venv/bin/python evaluate.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]
from stable_baselines3 import PPO

from baselines import (
    buy_and_hold_spy,
    equal_weight,
    evaluate_portfolio,
    momentum,
)
from data import TICKERS, load_data
from environment import PortfolioEnv

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Number of bootstrap resamples for the Sharpe CI
BOOTSTRAP_SAMPLES: int = 1000

# Rolling window (trading days) for the rolling Sharpe plot
ROLLING_WINDOW: int = 252

# Output resolution for saved plots
PLOT_DPI: int = 150

# Directory where plots are saved — resolves to <project_root>/runs/plots/
PLOTS_DIR: Path = Path(__file__).resolve().parent.parent / "runs" / "plots"

# Path to the best model saved by train.py
BEST_MODEL_PATH: str = str(Path(__file__).resolve().parent.parent / "runs" / "best_model" / "best_model")

# Annualisation factor — consistent with baselines.py and train.py
TRADING_DAYS_PER_YEAR: int = 252

# ---------------------------------------------------------------------------
# Logging setup
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

type FloatArray = npt.NDArray[np.float64]
type Float32Array = npt.NDArray[np.float32]


# ---------------------------------------------------------------------------
# Agent rollout on the test set
# ---------------------------------------------------------------------------


def run_agent_on_test(
    model: PPO,
    test_features: Float32Array,
    test_prices: Float32Array,
) -> tuple[FloatArray, FloatArray, int]:
    """
    Run a single deterministic rollout on the test environment.

    Mirrors the run_validation logic in train.py exactly, but on the test
    split.  The episode is forced to start at the earliest valid timestep
    (env._window) so the entire test period is covered.

    Alignment rule
    --------------
    weights_arr[i] earns the log return log(prices_arr[i] / prices_arr[i-1]).
    Row 0 is a dummy anchor; evaluate_portfolio skips it (it uses weights[1:]
    paired with log returns computed from prices[1:] / prices[:-1]).

    Parameters
    ----------
    model:
        Trained PPO model with a deterministic policy.
    test_features:
        Normalised feature array for the test split, shape (T_test, 15).
    test_prices:
        Raw adjusted close prices for the test split, shape (T_test, 5).

    Returns
    -------
    weights_arr:
        Portfolio weights over time, shape (T_episode, 5), dtype float64.
    prices_arr:
        Aligned price rows, shape (T_episode, 5), dtype float64.
    start_idx:
        The index into test_prices / test_dates where the episode begins
        (= env._window). Returned directly to avoid a fragile float-comparison
        scan of the price array in the caller.
    """
    N: int = test_prices.shape[1]
    env = PortfolioEnv(test_features, test_prices)

    # Force start at the earliest valid timestep, bypassing the random start
    obs, _ = env.reset(seed=0)
    env._t = env._window  # type: ignore[attr-defined]
    obs = env._get_obs()  # type: ignore[attr-defined]

    start_idx: int = env._window  # type: ignore[attr-defined]
    current_t: int = start_idx

    # Anchor row: the price and a dummy equal-weight entry at the starting
    # timestep.  evaluate_portfolio will not use this weight row in its return
    # calculation — it only uses weights[1:].
    step_prices: list[Float32Array] = [test_prices[current_t]]
    weights_list: list[Float32Array] = [
        np.full(N, 1.0 / N, dtype=np.float32)  # dummy anchor — not used
    ]

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        current_t += 1  # mirrors the _t increment inside env.step()
        weights_list.append(info["weights"])
        step_prices.append(test_prices[current_t])

    weights_arr: FloatArray = np.stack(weights_list).astype(np.float64)
    prices_arr: FloatArray = np.stack(step_prices).astype(np.float64)

    logger.info(
        f"Agent rollout complete: {len(weights_list)} steps "
        f"({weights_arr.shape[0] - 1} return observations)"
    )

    return weights_arr, prices_arr, start_idx


# ---------------------------------------------------------------------------
# Baseline weight helper
# ---------------------------------------------------------------------------


def compute_baseline_weights(
    prices: FloatArray,
    baseline_fn: Callable[..., FloatArray],
    **kwargs: object,
) -> FloatArray:
    """
    Call a baseline strategy function and return its weight array.

    Parameters
    ----------
    prices:
        Price array of shape (T, N).
    baseline_fn:
        One of the strategy functions from baselines.py (equal_weight,
        momentum).  Must accept prices as the first positional argument
        and return an (T, N) weight array.
    **kwargs:
        Additional keyword arguments forwarded to baseline_fn (e.g.
        lookback, rebalance_freq).

    Returns
    -------
    weights: FloatArray of shape (T, N).
    """
    return baseline_fn(prices, **kwargs)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Metrics table printing
# ---------------------------------------------------------------------------


def _print_metrics_table(metrics_by_strategy: dict[str, dict[str, float]]) -> None:
    """
    Print a formatted table of performance metrics to stdout.

    Rows are strategies; columns are the five standard metrics.
    """
    columns: list[str] = [
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
    ]

    # Column header labels (shorter for readability)
    col_labels: dict[str, str] = {
        "annualised_return": "Ann. Return",
        "annualised_volatility": "Ann. Vol",
        "sharpe_ratio": "Sharpe",
        "max_drawdown": "Max DD",
        "calmar_ratio": "Calmar",
    }

    strategy_col_width: int = max(len(s) for s in metrics_by_strategy) + 2
    col_width: int = 12

    header_parts = [f"{'Strategy':<{strategy_col_width}}"] + [
        f"{col_labels[c]:>{col_width}}" for c in columns
    ]
    header = "  ".join(header_parts)
    divider = "-" * len(header)

    print(f"\n{divider}")
    print(header)
    print(divider)

    for strategy, metrics in metrics_by_strategy.items():
        row_parts = [f"{strategy:<{strategy_col_width}}"]
        for col in columns:
            val = metrics[col]
            if col in ("annualised_return", "annualised_volatility", "max_drawdown"):
                # Format as percentage
                row_parts.append(f"{val * 100:>+{col_width - 1}.2f}%")
            elif col == "calmar_ratio" and val == float("inf"):
                row_parts.append(f"{'inf':>{col_width}}")
            else:
                row_parts.append(f"{val:>{col_width}.4f}")
        print("  ".join(row_parts))

    print(divider)


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------


def _bootstrap_sharpe_diff_ci(
    ppo_weights: FloatArray,
    ppo_prices: FloatArray,
    ew_weights: FloatArray,
    ew_prices: FloatArray,
    n_samples: int = BOOTSTRAP_SAMPLES,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """
    Compute a 95% bootstrap confidence interval for the Sharpe ratio
    difference between the PPO agent and the equal-weight baseline.

    We draw one bootstrap resample index per iteration and apply the SAME
    index to both strategies (paired bootstrap).  This preserves the temporal
    correlation between the two return series and correctly measures the
    distribution of their Sharpe difference.  The 2.5th and 97.5th percentiles
    form the 95% CI.

    Parameters
    ----------
    ppo_weights, ppo_prices:
        Agent weight and price arrays as returned by run_agent_on_test.
    ew_weights, ew_prices:
        Equal-weight strategy arrays of the same length.
    n_samples:
        Number of bootstrap resamples.
    rng_seed:
        Random seed for reproducibility.

    Returns
    -------
    (ci_lower, ci_upper): 95% CI bounds on Sharpe(PPO) - Sharpe(EqualWeight).
    """
    rng = np.random.default_rng(rng_seed)

    # Daily log returns for each strategy — shape (T-1,)
    # Weights[1:] earn log(prices[1:] / prices[:-1]), same convention as
    # evaluate_portfolio.
    def _daily_returns(weights: FloatArray, prices: FloatArray) -> FloatArray:
        asset_log_rets: FloatArray = np.log(prices[1:] / prices[:-1])
        return np.sum(weights[1:] * asset_log_rets, axis=1)

    ppo_daily: FloatArray = _daily_returns(ppo_weights, ppo_prices)
    ew_daily: FloatArray = _daily_returns(ew_weights, ew_prices)

    T: int = len(ppo_daily)

    def _sharpe(returns: FloatArray) -> float:
        """Annualised Sharpe from a 1-D daily log-return array."""
        std: float = float(returns.std(ddof=1))
        if std == 0.0:
            return 0.0
        return float(returns.mean()) / std * np.sqrt(TRADING_DAYS_PER_YEAR)

    diffs: FloatArray = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        idx: npt.NDArray[np.intp] = rng.integers(0, T, size=T)
        diffs[i] = _sharpe(ppo_daily[idx]) - _sharpe(ew_daily[idx])

    ci_lower: float = float(np.percentile(diffs, 2.5))
    ci_upper: float = float(np.percentile(diffs, 97.5))
    return ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Portfolio value helper (normalised to 1.0 at start)
# ---------------------------------------------------------------------------


def _portfolio_value_curve(weights: FloatArray, prices: FloatArray) -> FloatArray:
    """
    Compute a cumulative portfolio value curve starting at 1.0.

    Uses the same log-return convention as evaluate_portfolio: weights[i]
    earns log(prices[i] / prices[i-1]).

    Returns a 1-D array of shape (T,) where index 0 = 1.0 (the anchor).
    """
    asset_log_rets: FloatArray = np.log(prices[1:] / prices[:-1])
    port_log_rets: FloatArray = np.sum(weights[1:] * asset_log_rets, axis=1)
    # Prepend 0.0 so the curve length equals the number of price rows
    all_log_rets: FloatArray = np.concatenate([[0.0], port_log_rets])
    return np.exp(np.cumsum(all_log_rets))


# ---------------------------------------------------------------------------
# Rolling Sharpe helper
# ---------------------------------------------------------------------------


def _rolling_sharpe(
    weights: FloatArray, prices: FloatArray, window: int = ROLLING_WINDOW
) -> FloatArray:
    """
    Compute the rolling annualised Sharpe ratio over a sliding window.

    Returns a 1-D array of shape (T,).  Values in the first (window - 1)
    positions are NaN because the window is not yet fully populated.
    Index 0 of the returned array corresponds to the anchor row.
    """
    asset_log_rets: FloatArray = np.log(prices[1:] / prices[:-1])
    port_returns: FloatArray = np.sum(weights[1:] * asset_log_rets, axis=1)

    T_ret: int = len(port_returns)
    sharpes: FloatArray = np.full(T_ret, np.nan, dtype=np.float64)

    for t in range(window - 1, T_ret):
        window_rets = port_returns[t - window + 1 : t + 1]
        std: float = float(window_rets.std(ddof=1))
        if std > 0.0:
            sharpes[t] = (
                float(window_rets.mean()) / std * np.sqrt(TRADING_DAYS_PER_YEAR)
            )

    # Prepend NaN for the anchor row so length == len(prices)
    return np.concatenate([[np.nan], sharpes])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_portfolio_value(
    curves: dict[str, FloatArray],
    dates: pd.DatetimeIndex,
    output_path: Path,
) -> None:
    """
    Save a line chart of normalised portfolio values over time.

    Parameters
    ----------
    curves:
        Dict mapping strategy name -> 1-D equity curve array of shape (T,).
    dates:
        DatetimeIndex of length T aligned with the equity curves.
    output_path:
        File path to write the PNG.
    """
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, curve in curves.items():
        ax.plot(dates, curve, label=name, linewidth=1.5)

    ax.set_title("Portfolio Value Over Time (normalised to 1.0 at start)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def _plot_rolling_sharpe(
    rolling_sharpes: dict[str, FloatArray],
    dates: pd.DatetimeIndex,
    output_path: Path,
) -> None:
    """
    Save a line chart of rolling Sharpe ratios over time.

    Parameters
    ----------
    rolling_sharpes:
        Dict mapping strategy name -> 1-D rolling Sharpe array of shape (T,).
    dates:
        DatetimeIndex of length T.
    output_path:
        File path to write the PNG.
    """
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, rs in rolling_sharpes.items():
        ax.plot(dates, rs, label=name, linewidth=1.5)

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(f"{ROLLING_WINDOW}-Day Rolling Sharpe Ratio")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Sharpe Ratio (annualised)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def _plot_weight_heatmap(
    weights: FloatArray,
    dates: pd.DatetimeIndex,
    asset_names: list[str],
    output_path: Path,
) -> None:
    """
    Save a heatmap of PPO agent portfolio weights over time.

    The heatmap has time on the x-axis and asset on the y-axis; colour
    encodes the weight (0 to 1).

    Parameters
    ----------
    weights:
        Weight array of shape (T, N).  Row 0 is the dummy anchor and is
        excluded from the plot.
    dates:
        DatetimeIndex of length T.  dates[1:] is used to skip the anchor.
    asset_names:
        List of N asset ticker strings used as y-axis labels.
    output_path:
        File path to write the PNG.
    """
    # Exclude the anchor row (row 0) — it is a dummy and not earned
    plot_weights: FloatArray = weights[1:].T  # shape (N, T-1)
    plot_dates = dates[1:]

    # Subsample columns for a readable heatmap if the series is long
    # (seaborn heatmap renders every column — thin to at most 252 columns)
    max_cols: int = 252
    if plot_weights.shape[1] > max_cols:
        step: int = plot_weights.shape[1] // max_cols
        plot_weights = plot_weights[:, ::step]
        plot_dates = plot_dates[::step]

    # Build a DataFrame with ticker labels for the y-axis
    df = pd.DataFrame(
        plot_weights,
        index=asset_names,
        columns=[d.strftime("%Y-%m") for d in plot_dates],
    )

    sns.set_theme(style="white")
    # Height scales with number of assets; width with time coverage
    fig_width: float = min(max(10.0, len(df.columns) / 10), 20.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4))

    sns.heatmap(
        df,
        ax=ax,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.0,
        cbar_kws={"label": "Weight"},
    )

    ax.set_title("PPO Agent Portfolio Weights Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Asset")

    # Show only a manageable number of x-tick labels to avoid overlap
    n_ticks: int = 10
    tick_positions = np.linspace(0, len(df.columns) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([df.columns[i] for i in tick_positions], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Full test-set evaluation pipeline:

    1. Load data (test split only)
    2. Load best PPO checkpoint
    3. Run agent deterministic rollout on the test set
    4. Compute baseline weights (equal-weight, momentum)
    5. Compute SPY buy-and-hold equity curve
    6. Print metrics table
    7. Bootstrap 95% CI on Sharpe difference vs equal-weight baseline
    8. Save three plots to plots/
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data — test split only
    # ------------------------------------------------------------------
    logger.info("Loading data ...")
    data = load_data()

    test_features: Float32Array = data["test"]
    test_prices: Float32Array = data["test_prices"]
    test_spy: Float32Array = data["spy_test"]
    test_dates: pd.DatetimeIndex = data["test_dates"]

    # ------------------------------------------------------------------
    # 2. Load the best model checkpoint
    # ------------------------------------------------------------------
    logger.info(f"Loading model from {BEST_MODEL_PATH} ...")
    model: PPO = PPO.load(BEST_MODEL_PATH)

    # ------------------------------------------------------------------
    # 3. Run agent on the test set
    # ------------------------------------------------------------------
    logger.info("Running agent rollout on test set ...")
    ppo_weights, ppo_prices, start_idx = run_agent_on_test(model, test_features, test_prices)

    episode_len: int = ppo_prices.shape[0]
    episode_dates: pd.DatetimeIndex = test_dates[start_idx : start_idx + episode_len]

    # ------------------------------------------------------------------
    # 4. Compute baseline weights on the same price slice as the agent
    # ------------------------------------------------------------------
    # Use the same price rows the agent operated on so all arrays are
    # aligned (same T, same dates).
    slice_prices: FloatArray = ppo_prices  # already float64

    ew_weights: FloatArray = compute_baseline_weights(
        slice_prices, equal_weight  # type: ignore[arg-type]
    )
    mom_weights: FloatArray = compute_baseline_weights(
        slice_prices, momentum  # type: ignore[arg-type]
    )

    # ------------------------------------------------------------------
    # 5. SPY equity curve — also sliced to the episode length
    # ------------------------------------------------------------------
    spy_slice: FloatArray = test_spy[start_idx : start_idx + episode_len].astype(
        np.float64
    )
    spy_curve: FloatArray = buy_and_hold_spy(spy_slice)

    # ------------------------------------------------------------------
    # 6. Compute metrics for each strategy
    # ------------------------------------------------------------------
    logger.info("Computing metrics ...")
    metrics_ppo = evaluate_portfolio(ppo_prices, ppo_weights)
    metrics_ew = evaluate_portfolio(slice_prices, ew_weights)
    metrics_mom = evaluate_portfolio(slice_prices, mom_weights)

    # SPY metrics: buy_and_hold_spy returns an equity curve, not weights.
    # Reconstruct a dummy single-asset price/weight pair to feed into
    # evaluate_portfolio: shape (T, 1) prices, weight always 1.0.
    spy_prices_2d: FloatArray = spy_slice.reshape(-1, 1)
    spy_weights_2d: FloatArray = np.ones((episode_len, 1), dtype=np.float64)
    metrics_spy = evaluate_portfolio(spy_prices_2d, spy_weights_2d)

    metrics_by_strategy: dict[str, dict[str, float]] = {
        "PPO Agent": metrics_ppo,
        "Equal Weight": metrics_ew,
        "Momentum": metrics_mom,
        "SPY (Buy & Hold)": metrics_spy,
    }

    # ------------------------------------------------------------------
    # 7. Print metrics table
    # ------------------------------------------------------------------
    print("\n=== Test Set Performance Metrics ===")
    _print_metrics_table(metrics_by_strategy)

    # ------------------------------------------------------------------
    # 7b. Bootstrap CI on Sharpe difference: PPO vs Equal Weight
    # ------------------------------------------------------------------
    logger.info(f"Bootstrapping Sharpe CI ({BOOTSTRAP_SAMPLES} samples) ...")
    ci_lower, ci_upper = _bootstrap_sharpe_diff_ci(
        ppo_weights, ppo_prices, ew_weights, slice_prices
    )
    print(
        f"\nBootstrap 95% CI for Sharpe(PPO) - Sharpe(EqualWeight): "
        f"[{ci_lower:+.4f}, {ci_upper:+.4f}]"
        f"  (n={BOOTSTRAP_SAMPLES} resamples)"
    )
    if ci_lower > 0.0:
        print("  -> PPO Sharpe is significantly HIGHER than equal-weight (p < 0.05).")
    elif ci_upper < 0.0:
        print("  -> PPO Sharpe is significantly LOWER than equal-weight (p < 0.05).")
    else:
        print("  -> Difference is NOT statistically significant at the 5% level.")

    # ------------------------------------------------------------------
    # 8. Generate and save plots
    # ------------------------------------------------------------------
    logger.info("Generating plots ...")

    # --- 8a. Portfolio value over time ---
    ppo_curve: FloatArray = _portfolio_value_curve(ppo_weights, ppo_prices)
    ew_curve: FloatArray = _portfolio_value_curve(ew_weights, slice_prices)
    mom_curve: FloatArray = _portfolio_value_curve(mom_weights, slice_prices)

    _plot_portfolio_value(
        curves={
            "PPO Agent": ppo_curve,
            "Equal Weight": ew_curve,
            "Momentum": mom_curve,
            "SPY (Buy & Hold)": spy_curve,
        },
        dates=episode_dates,
        output_path=PLOTS_DIR / "portfolio_value.png",
    )

    # --- 8b. Rolling Sharpe ratio ---
    _plot_rolling_sharpe(
        rolling_sharpes={
            "PPO Agent": _rolling_sharpe(ppo_weights, ppo_prices),
            "Equal Weight": _rolling_sharpe(ew_weights, slice_prices),
            "Momentum": _rolling_sharpe(mom_weights, slice_prices),
        },
        dates=episode_dates,
        output_path=PLOTS_DIR / "rolling_sharpe.png",
    )

    # --- 8c. PPO weight heatmap ---
    _plot_weight_heatmap(
        weights=ppo_weights,
        dates=episode_dates,
        asset_names=TICKERS,
        output_path=PLOTS_DIR / "weight_heatmap.png",
    )

    print(f"\nPlots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
