"""
data.py — Data ingestion and feature engineering for rl-portfolio-agent.

Downloads adjusted close prices for five sector ETFs (and SPY as a baseline
reference) via yfinance, computes three features per asset, normalises using
training statistics only, and returns numpy arrays for the train / val / test
splits plus raw prices and dates for downstream use.

Usage:
    from data import load_data
    data = load_data()
"""

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants — change these to retrain on a different universe or time window
# ---------------------------------------------------------------------------

# The five sector ETFs the agent will allocate across
TICKERS = ["XLK", "XLE", "XLF", "XLV", "XLI"]

# SPY is downloaded separately and used only for baseline comparisons
SPY = "SPY"

# Rolling window (in trading days) for volatility and mean-reversion features
WINDOW = 20

# Momentum lookback windows (in trading days): ~3 months and ~12 months
MOM_SHORT = 63
MOM_LONG = 252

# Split boundaries (inclusive start, exclusive end — standard Pandas convention)
TRAIN_START = "2000-01-01"
TRAIN_END = "2015-01-01"  # train: [TRAIN_START, TRAIN_END)
VAL_START = "2015-01-01"
VAL_END = "2019-01-01"  # val:   [VAL_START, VAL_END)
TEST_START = "2019-01-01"
TEST_END = "2025-01-01"  # test:  [TEST_START, TEST_END)


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------


def _log_returns(prices: pd.Series) -> pd.Series:
    """
    Log return at time t: log(P_t / P_{t-1}).

    Log returns are preferred over simple returns because:
    - They are additive over time (multi-period return = sum of daily log returns)
    - They are approximately normally distributed for small moves
    - They avoid the asymmetry between gains and losses in simple returns
    """
    return np.log(prices / prices.shift(1))


def _rolling_volatility(log_rets: pd.Series, window: int) -> pd.Series:
    """
    Rolling standard deviation of log returns over `window` trading days.

    This is the most common empirical volatility estimator. A window of 20
    trading days corresponds roughly to one calendar month.
    """
    return log_rets.rolling(window=window).std()


def _momentum_return(prices: pd.Series, window: int) -> pd.Series:
    """
    Cumulative log return over `window` trading days: log(P_t / P_{t-window}).

    Captures medium- and long-horizon price momentum. Positive values indicate
    the asset has appreciated over the lookback; negative values indicate decline.
    """
    return np.log(prices / prices.shift(window))


def _mean_reversion_signal(prices: pd.Series, window: int) -> pd.Series:
    """
    Z-score of price relative to its rolling mean and std:
        z_t = (P_t - rolling_mean_t) / rolling_std_t

    A positive z-score means the price is above its recent average (potentially
    overbought); a negative z-score means it is below (potentially oversold).
    This gives the agent a stationary signal about where each asset sits
    relative to its recent trend.
    """
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    return (prices - rolling_mean) / rolling_std


def _build_feature_matrix(prices_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Build the full (T × 5N) feature matrix from a (T × N) price DataFrame.

    Column order: [asset0_logret, asset0_vol, asset0_meanrev,
                   asset0_mom63, asset0_mom252,
                   asset1_logret, ...]

    The first MOM_LONG rows will contain NaNs from the momentum lookback;
    these are dropped by the caller after the three splits are assembled.
    """
    columns = {}
    for ticker in prices_df.columns:
        p = prices_df[ticker]
        log_ret = _log_returns(p)
        vol = _rolling_volatility(log_ret, window)
        meanrev = _mean_reversion_signal(p, window)
        mom63 = _momentum_return(p, MOM_SHORT)
        mom252 = _momentum_return(p, MOM_LONG)

        columns[f"{ticker}_logret"] = log_ret
        columns[f"{ticker}_vol"] = vol
        columns[f"{ticker}_meanrev"] = meanrev
        columns[f"{ticker}_mom63"] = mom63
        columns[f"{ticker}_mom252"] = mom252

    return pd.DataFrame(columns, index=prices_df.index)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def load_data() -> dict:
    """
    Download prices, engineer features, normalise, and return split arrays.

    Returns a dict with the following keys:
        train / val / test          — normalised feature arrays (T, 15)
        feature_names               — list of 15 feature name strings
        train_prices / val_prices / test_prices
                                    — raw adjusted close prices (T, 5)
        spy_train / spy_val / spy_test
                                    — SPY adjusted close as a 1-D array
        train_dates / val_dates / test_dates
                                    — pd.DatetimeIndex for each split
    """

    # ------------------------------------------------------------------
    # 1. Download price data
    #
    # yfinance returns a DataFrame indexed by date. We download the full
    # range in one call to avoid repeated network requests. auto_adjust=True
    # returns the split- and dividend-adjusted close directly in the
    # 'Close' column (no separate 'Adj Close' column needed).
    # ------------------------------------------------------------------
    all_tickers = TICKERS + [SPY]

    print(f"Downloading price data for: {all_tickers}")
    raw = yf.download(
        tickers=all_tickers,
        start=TRAIN_START,
        end=TEST_END,
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns a MultiIndex when multiple tickers are requested.
    # We only need the 'Close' level (which is the adjusted close when
    # auto_adjust=True).
    prices_all = raw["Close"]

    # Drop any dates where all tickers are NaN (e.g. market holidays that
    # yfinance may include at boundaries), then forward-fill the remaining
    # isolated NaNs (e.g. a single ticker missing one day due to data issues).
    prices_all = prices_all.dropna(how="all").ffill(limit=5)
    # Fail loudly if any column still has NaNs after forward-filling — a limit=5
    # gap means a ticker was absent for more than a week, which is not safe to ignore.
    assert not prices_all.isnull().any().any(), (
        f"NaNs remain after ffill(limit=5). Affected columns: "
        f"{prices_all.columns[prices_all.isnull().any()].tolist()}"
    )

    # Separate the sector ETFs from SPY
    sector_prices = prices_all[TICKERS]
    spy_prices = prices_all[SPY]

    # ------------------------------------------------------------------
    # 2. Compute features across the full date range
    #
    # We compute features on the full price history before splitting so
    # that the rolling windows at split boundaries are correct. A row that
    # straddles a split would otherwise have an artificially short window.
    # ------------------------------------------------------------------
    features_all = _build_feature_matrix(sector_prices, WINDOW)

    # ------------------------------------------------------------------
    # 3. Split into train / val / test by date
    # ------------------------------------------------------------------
    def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        """Return rows in [start, end) inclusive of start, exclusive of end."""
        return df[(df.index >= start) & (df.index < end)]

    feat_train = _slice(features_all, TRAIN_START, TRAIN_END)
    feat_val = _slice(features_all, VAL_START, VAL_END)
    feat_test = _slice(features_all, TEST_START, TEST_END)

    prices_train = _slice(sector_prices, TRAIN_START, TRAIN_END)
    prices_val = _slice(sector_prices, VAL_START, VAL_END)
    prices_test = _slice(sector_prices, TEST_START, TEST_END)

    spy_train_raw = _slice(spy_prices.to_frame(), TRAIN_START, TRAIN_END)[SPY]
    spy_val_raw = _slice(spy_prices.to_frame(), VAL_START, VAL_END)[SPY]
    spy_test_raw = _slice(spy_prices.to_frame(), TEST_START, TEST_END)[SPY]

    # ------------------------------------------------------------------
    # 4. Drop burn-in rows caused by the rolling window
    #
    # The rolling volatility feature uses log returns as input, so it needs
    # WINDOW observations of log returns, each of which itself requires one
    # prior price. This means the effective burn-in is WINDOW + 1 rows.
    # Val and test should not have burn-in NaNs (their windows are
    # satisfied by price history from before the split boundary), but we
    # verify this via the assertion below.
    # ------------------------------------------------------------------
    rows_before = len(feat_train)
    feat_train = feat_train.dropna()
    rows_dropped = rows_before - len(feat_train)
    assert rows_dropped <= MOM_LONG + 2, (
        f"dropna() dropped {rows_dropped} rows from train — expected at most "
        f"{MOM_LONG + 2}. Possible mid-series NaNs in the downloaded data."
    )
    prices_train = prices_train.loc[feat_train.index]
    spy_train_raw = spy_train_raw.loc[feat_train.index]

    # Val and test: drop any residual NaNs (should be zero rows dropped)
    feat_val = feat_val.dropna()
    feat_test = feat_test.dropna()
    prices_val = prices_val.loc[feat_val.index]
    prices_test = prices_test.loc[feat_test.index]
    spy_val_raw = spy_val_raw.loc[feat_val.index]
    spy_test_raw = spy_test_raw.loc[feat_test.index]

    # ------------------------------------------------------------------
    # 5. Normalise features using training statistics only
    #
    # We compute the mean and std of each feature column on the training
    # set and apply the same transformation to val and test. Using training
    # stats on all splits prevents data leakage: the model never "sees"
    # future distribution information during training or evaluation.
    # ------------------------------------------------------------------
    train_mean = feat_train.mean()  # shape (n_features,)
    train_std = feat_train.std()  # shape (n_features,)

    # Guard against zero-variance features (would cause division by zero or
    # numerically degenerate values for subnormal std). Clip to a small floor
    # rather than checking for exact 0.0 only.
    train_std = train_std.clip(lower=1e-8)

    feat_train_norm = (feat_train - train_mean) / train_std
    feat_val_norm = (feat_val - train_mean) / train_std
    feat_test_norm = (feat_test - train_mean) / train_std

    # ------------------------------------------------------------------
    # 6. Convert to numpy arrays
    # ------------------------------------------------------------------
    train_arr = feat_train_norm.to_numpy(dtype=np.float32)
    val_arr = feat_val_norm.to_numpy(dtype=np.float32)
    test_arr = feat_test_norm.to_numpy(dtype=np.float32)

    train_prices_arr = prices_train.to_numpy(dtype=np.float32)
    val_prices_arr = prices_val.to_numpy(dtype=np.float32)
    test_prices_arr = prices_test.to_numpy(dtype=np.float32)

    spy_train_arr = spy_train_raw.to_numpy(dtype=np.float32)
    spy_val_arr = spy_val_raw.to_numpy(dtype=np.float32)
    spy_test_arr = spy_test_raw.to_numpy(dtype=np.float32)

    feature_names = list(feat_train.columns)

    # ------------------------------------------------------------------
    # 7. Sanity assertions
    # ------------------------------------------------------------------

    # No NaNs in any output array after burn-in drop
    assert not np.isnan(train_arr).any(), "NaNs found in train feature array"
    assert not np.isnan(val_arr).any(), "NaNs found in val feature array"
    assert not np.isnan(test_arr).any(), "NaNs found in test feature array"

    # Non-overlapping date ranges: train ends before val starts, val ends
    # before test starts (using the boundary dates, not actual trading days)
    assert (
        feat_train.index.max() < feat_val.index.min()
    ), "Train and val date ranges overlap"
    assert (
        feat_val.index.max() < feat_test.index.min()
    ), "Val and test date ranges overlap"

    # Normalisation stats were computed only on train data: verify the
    # mean and std Series have one entry per feature column (not inflated
    # by including val/test rows).
    assert len(train_mean) == len(
        feature_names
    ), "train_mean length does not match number of features"
    assert len(train_std) == len(
        feature_names
    ), "train_std length does not match number of features"

    # Feature matrix should have 5 features × N tickers columns
    n_expected_cols = 5 * len(TICKERS)
    assert (
        train_arr.shape[1] == n_expected_cols
    ), f"Expected {n_expected_cols} feature columns, got {train_arr.shape[1]}"

    print(
        f"Train: {train_arr.shape} rows  ({feat_train.index[0].date()} – {feat_train.index[-1].date()})"
    )
    print(
        f"Val:   {val_arr.shape} rows  ({feat_val.index[0].date()} – {feat_val.index[-1].date()})"
    )
    print(
        f"Test:  {test_arr.shape} rows  ({feat_test.index[0].date()} – {feat_test.index[-1].date()})"
    )

    return {
        "train": train_arr,
        "val": val_arr,
        "test": test_arr,
        "feature_names": feature_names,
        "train_prices": train_prices_arr,
        "val_prices": val_prices_arr,
        "test_prices": test_prices_arr,
        "spy_train": spy_train_arr,
        "spy_val": spy_val_arr,
        "spy_test": spy_test_arr,
        "train_dates": feat_train.index,
        "val_dates": feat_val.index,
        "test_dates": feat_test.index,
    }


if __name__ == "__main__":
    data = load_data()
    print("\nFeature names:", data["feature_names"])
    print("Train shape:", data["train"].shape)
    print("Val shape:  ", data["val"].shape)
    print("Test shape: ", data["test"].shape)
