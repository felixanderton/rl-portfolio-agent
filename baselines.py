"""
baselines.py — Benchmark strategies for rl-portfolio-agent.

Provides three baseline allocation strategies (equal weight, cross-sectional
momentum, buy-and-hold SPY) and a shared evaluation function that computes
standard performance metrics from a price series and a weights history.

These baselines establish a performance floor against which the RL agent is
compared in evaluate.py.

Usage:
    import numpy as np
    from baselines import equal_weight, momentum, buy_and_hold_spy, evaluate_portfolio

    weights = equal_weight(prices)          # (T, N) weight array
    metrics = evaluate_portfolio(prices, weights)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

type FloatArray = npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of trading days assumed in a year — standard US equity convention.
TRADING_DAYS_PER_YEAR: int = 252

# Risk-free rate used in the Sharpe ratio calculation (0 = excess-return Sharpe).
RISK_FREE_RATE: float = 0.0


# ---------------------------------------------------------------------------
# Shared evaluation function
# ---------------------------------------------------------------------------


def evaluate_portfolio(
    prices: npt.NDArray[np.floating],
    weights_over_time: npt.NDArray[np.floating],
) -> dict[str, float]:
    """
    Compute standard performance metrics for a strategy described by
    a weight history applied to a set of asset prices.

    Parameters
    ----------
    prices:
        Adjusted close price array of shape (T, N).
        Row 0 is the earliest date; prices must be strictly positive.
    weights_over_time:
        Portfolio weight array of shape (T, N).
        Row t contains the weights that were held *into* period t, earning
        the return from prices[t-1] to prices[t].
        Row 0 is unused (there is no prior period to earn a return from).

    Returns
    -------
    dict with keys:
        annualised_return       — geometric annualised return (decimal)
        annualised_volatility   — annualised return standard deviation
        sharpe_ratio            — annualised return / annualised volatility
        max_drawdown            — maximum peak-to-trough drawdown (negative or 0)
        calmar_ratio            — annualised return / |max_drawdown| (0 if no drawdown)
    """
    prices = np.asarray(prices, dtype=np.float64)
    weights_over_time = np.asarray(weights_over_time, dtype=np.float64)

    T, N = prices.shape
    assert weights_over_time.shape == (T, N), (
        f"weights_over_time shape {weights_over_time.shape} does not match "
        f"prices shape {prices.shape}"
    )

    # ------------------------------------------------------------------
    # 1. Daily portfolio log returns
    #
    # Log return for asset i at time t: log(P_t[i] / P_{t-1}[i]).
    # Portfolio log return at time t: sum over i of w_t[i] * log_ret_t[i].
    #
    # We use row t's weights with the return earned *between* t-1 and t,
    # so the series has T-1 valid values (indices 1 through T-1).
    # ------------------------------------------------------------------
    # Shape: (T-1, N) — log return for each asset from day t-1 to day t
    asset_log_returns: FloatArray = np.log(prices[1:] / prices[:-1])

    # Weights applied to earn those returns: rows 1 through T-1.
    # Cast explicitly to float64 to satisfy mypy — weights_over_time was
    # already converted via np.asarray(..., dtype=np.float64) above, so this
    # is a no-op at runtime.
    w: FloatArray = weights_over_time[1:].astype(np.float64)

    # Portfolio return at each step: dot product of weights and log returns
    portfolio_returns: FloatArray = np.sum(
        w * asset_log_returns, axis=1
    )  # shape (T-1,)

    # ------------------------------------------------------------------
    # 2. Annualised return
    #
    # Geometric annualisation converts the mean daily log return to the
    # equivalent constant daily compound rate, then scales to one year:
    #   annualised_return = exp(mean(R) * 252) - 1
    # This is exact for log returns: summing daily log returns gives the
    # total log return, and exponentiation converts back to a growth factor.
    # ------------------------------------------------------------------
    mean_daily_log_return: float = float(portfolio_returns.mean())
    annualised_return: float = float(
        np.exp(mean_daily_log_return * TRADING_DAYS_PER_YEAR) - 1.0
    )

    # ------------------------------------------------------------------
    # 3. Annualised volatility
    #
    # Standard deviation of daily log returns (ddof=1 for sample estimate)
    # scaled by sqrt(252). This matches the rolling volatility convention
    # used in data.py.
    # ------------------------------------------------------------------
    daily_vol: float = float(portfolio_returns.std(ddof=1))
    annualised_volatility: float = daily_vol * float(np.sqrt(TRADING_DAYS_PER_YEAR))

    # ------------------------------------------------------------------
    # 4. Sharpe ratio
    #
    # Annualised excess return divided by annualised volatility.
    # With a zero risk-free rate this equals annualised_return / annualised_vol.
    # Guard against zero volatility (e.g. a flat price series in testing).
    # ------------------------------------------------------------------
    excess_return: float = annualised_return - RISK_FREE_RATE
    if annualised_volatility == 0.0:
        sharpe_ratio: float = 0.0
    else:
        sharpe_ratio = excess_return / annualised_volatility

    # ------------------------------------------------------------------
    # 5. Maximum drawdown
    #
    # The maximum peak-to-trough decline in the cumulative portfolio value.
    # We start with a portfolio value of 1.0 and grow it by the realised
    # log returns: value_t = exp(cumsum(R)_t).
    #
    # At each time step we compare the current value to the highest value
    # seen so far (the "high-water mark"). The drawdown at time t is:
    #   drawdown_t = (value_t - peak_t) / peak_t
    # which is always <= 0. The max drawdown is the minimum drawdown.
    # ------------------------------------------------------------------
    # Cumulative portfolio value (starting at 1.0 before the first return).
    # We prepend 1.0 so the high-water mark is always initialised to the
    # starting value; without it, a negative first return would set the
    # initial peak below 1.0 and understate all subsequent drawdowns.
    cumulative_value: FloatArray = np.concatenate(
        [[1.0], np.exp(np.cumsum(portfolio_returns))]
    )

    # Running maximum (high-water mark) at each time step
    running_max: FloatArray = np.maximum.accumulate(cumulative_value)

    # Drawdown at each step — always <= 0
    drawdown: FloatArray = (cumulative_value - running_max) / running_max

    max_drawdown: float = float(drawdown.min())

    # ------------------------------------------------------------------
    # 6. Calmar ratio
    #
    # Annualised return divided by the absolute value of the maximum drawdown.
    # A higher Calmar ratio means more return per unit of peak-to-trough risk.
    # Return 0 if there is no drawdown (perfectly flat or always rising equity).
    # ------------------------------------------------------------------
    if max_drawdown == 0.0:
        # No drawdown occurred — equity curve was monotonically non-decreasing.
        # Return +inf to signal "unbounded" Calmar rather than 0, which would
        # incorrectly rank a perfect curve below one with any drawdown.
        calmar_ratio: float = float("inf")
    else:
        calmar_ratio = annualised_return / abs(max_drawdown)

    return {
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
    }


# ---------------------------------------------------------------------------
# Baseline 1: Equal Weight
# ---------------------------------------------------------------------------


def equal_weight(
    prices: npt.NDArray[np.floating],
    rebalance_freq: int = 21,
) -> npt.NDArray[np.float64]:
    """
    Equal-weight strategy: allocate 1/N to every asset, rebalancing
    every `rebalance_freq` trading days (approximately monthly at the default).

    Between rebalance days the weights are held constant — no drift
    adjustment is applied. This models a simple fixed-schedule rebalance
    without intra-period tracking.

    Parameters
    ----------
    prices:
        Price array of shape (T, N). Only the shape is used here;
        prices are not needed to compute equal weights.
    rebalance_freq:
        Number of trading days between rebalances. Day 0 is always a
        rebalance day.

    Returns
    -------
    weights: np.ndarray of shape (T, N) and dtype float64.
    """
    prices = np.asarray(prices, dtype=np.float64)
    T, N = prices.shape

    # Equal weight for one period: every asset gets the same allocation
    equal: npt.NDArray[np.float64] = np.full(N, 1.0 / N, dtype=np.float64)

    weights: npt.NDArray[np.float64] = np.empty((T, N), dtype=np.float64)

    # Track the most recently set weight vector — starts at equal weight
    current_weights: npt.NDArray[np.float64] = equal.copy()

    for t in range(T):
        # Rebalance on day 0 and every rebalance_freq days thereafter
        if t % rebalance_freq == 0:
            current_weights = equal.copy()
        weights[t] = current_weights

    return weights


# ---------------------------------------------------------------------------
# Baseline 2: Cross-sectional momentum (12-1 month)
# ---------------------------------------------------------------------------


def momentum(
    prices: npt.NDArray[np.floating],
    lookback: int = 252,
    skip: int = 21,
    rebalance_freq: int = 21,
) -> npt.NDArray[np.float64]:
    """
    Cross-sectional momentum strategy (12-1 month variant).

    On each rebalance day, rank assets by their return over the window
    [t - lookback, t - skip]. This skips the most recent `skip` days to
    avoid the short-term reversal effect documented in the academic
    literature (Jegadeesh & Titman, 1993).

    Assets with negative momentum receive a weight of zero before
    normalisation, concentrating the portfolio in positive-momentum names.
    If every asset has negative momentum, fall back to equal weight for
    that period to avoid holding cash (the agent always stays fully invested).

    Parameters
    ----------
    prices:
        Price array of shape (T, N), adjusted close.
    lookback:
        Number of trading days defining the start of the momentum window.
        Default 252 ≈ 12 months.
    skip:
        Number of most-recent trading days to exclude from the momentum
        calculation. Default 21 ≈ 1 month.
    rebalance_freq:
        Number of trading days between rebalances.

    Returns
    -------
    weights: np.ndarray of shape (T, N) and dtype float64.
    """
    prices = np.asarray(prices, dtype=np.float64)
    T, N = prices.shape

    equal: npt.NDArray[np.float64] = np.full(N, 1.0 / N, dtype=np.float64)
    weights: npt.NDArray[np.float64] = np.empty((T, N), dtype=np.float64)

    # Start with equal weight; updated on each rebalance day
    current_weights: npt.NDArray[np.float64] = equal.copy()

    for t in range(T):
        if t % rebalance_freq == 0:
            # Use equal weight whenever the full lookback window is not yet
            # available (early rows before we have `lookback` days of history).
            if t < lookback:
                current_weights = equal.copy()
            else:
                # Momentum signal: simple return from the start of the window
                # to the end, skipping the most recent `skip` days.
                #   start_idx = t - lookback
                #   end_idx   = t - skip
                # Return = price[end_idx] / price[start_idx] - 1
                start_prices: npt.NDArray[np.float64] = prices[
                    t - lookback
                ]  # shape (N,)
                end_prices: npt.NDArray[np.float64] = prices[t - skip]  # shape (N,)
                mom_signal: npt.NDArray[np.float64] = end_prices / start_prices - 1.0

                # Zero out assets with negative momentum before normalisation
                pos_momentum: npt.NDArray[np.float64] = np.where(
                    mom_signal > 0.0, mom_signal, 0.0
                )

                total_positive: float = float(pos_momentum.sum())

                if total_positive == 0.0:
                    # All assets have non-positive momentum — fall back to equal weight
                    current_weights = equal.copy()
                else:
                    # Allocate proportionally to positive momentum scores
                    current_weights = pos_momentum / total_positive

        weights[t] = current_weights

    return weights


# ---------------------------------------------------------------------------
# Baseline 3: Buy and hold SPY
# ---------------------------------------------------------------------------


def buy_and_hold_spy(spy_prices: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """
    Compute the cumulative value of a buy-and-hold position in SPY.

    This is not a weights array — it is a single equity curve starting at
    1.0, used as a visual and metric reference in evaluate.py.

    Parameters
    ----------
    spy_prices:
        1-D array of SPY adjusted close prices, shape (T,).

    Returns
    -------
    equity_curve: 1-D np.ndarray of shape (T,) and dtype float64.
        equity_curve[0] = 1.0 (initial value before any return is earned).
        equity_curve[t] = compounded value after t periods.
    """
    spy_prices = np.asarray(spy_prices, dtype=np.float64)
    T: int = spy_prices.shape[0]

    # Daily log returns: undefined for t=0 (no prior price), so we prepend 0.0
    # to keep the output length equal to T.
    log_returns: npt.NDArray[np.float64] = np.empty(T, dtype=np.float64)
    log_returns[0] = 0.0  # no return on the first day — value stays at 1.0
    log_returns[1:] = np.log(spy_prices[1:] / spy_prices[:-1])

    # Cumulative product of daily growth factors = exp of cumulative log returns.
    # Starting value is 1.0; cumsum(log_returns) at index t gives the total
    # log return from day 0 to day t.
    equity_curve: npt.NDArray[np.float64] = np.exp(np.cumsum(log_returns))

    return equity_curve
