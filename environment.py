"""
environment.py — Custom Gymnasium environment for portfolio allocation.

Wraps a normalised feature array (output of data.load_data()) in a Gymnasium
interface suitable for training a PPO agent via stable-baselines3.

Key design decisions:
- Action space: raw logits in [0, 1]^5; softmax is applied *inside* step() so
  the agent never has to learn a normalisation constraint itself.
- Reward: differential Sharpe ratio (Moody & Saffell 1998) net of transaction
  costs. This gives a dense, risk-adjusted signal at every timestep rather than
  a sparse end-of-episode return.
- Episode start: sampled uniformly from [window, T-1] so every episode begins
  with a full look-back window and episodes have varying lengths.

Usage:
    from data import load_data
    from environment import PortfolioEnv

    data = load_data()
    env = PortfolioEnv(features=data["train"], prices=data["train_prices"])
    obs, info = env.reset()
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import gymnasium  # type: ignore[import-untyped]
from gymnasium import spaces  # type: ignore[import-untyped]
from typing import Any, override

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

type FloatArray = npt.NDArray[np.float32]
type StepReturn = tuple[FloatArray, float, bool, bool, dict[str, Any]]
type ResetReturn = tuple[FloatArray, dict[str, Any]]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class PortfolioEnv(gymnasium.Env):
    """
    A Gymnasium environment for portfolio allocation across N assets.

    The agent observes a state vector built from:
      - Per-asset log returns over the past `window` days   (window × N values)
      - Rolling volatility of each asset at the current step (N values)
      - Mean-reversion signal for each asset at the current step (N values)
      - Current portfolio weights                            (N values)
      - Current portfolio volatility (scalar)               (1 value)

    Total observation size: window*N + N + N + N + 1

    The agent emits raw logits (floats in [0, 1]) for each asset; the
    environment normalises them via softmax before applying them as weights.
    """

    # ------------------------------------------------------------------
    # Class-level constants
    # ------------------------------------------------------------------

    # Number of assets the agent allocates across
    N_ASSETS: int = 5

    # Number of features per asset in the feature matrix (logret, vol, meanrev)
    N_FEATURES_PER_ASSET: int = 3

    # Column strides within the feature matrix for each per-asset feature.
    # data.py lays columns out as:
    #   [asset0_logret, asset0_vol, asset0_meanrev,
    #    asset1_logret, asset1_vol, asset1_meanrev, ...]
    # So for asset i: logret is at column i*3+0, vol at i*3+1, meanrev at i*3+2
    LOGRET_OFFSET: int = 0
    VOL_OFFSET: int = 1
    MEANREV_OFFSET: int = 2

    # Small epsilon to prevent division by zero in the Sharpe denominator
    SHARPE_EPS: float = 1e-8

    def __init__(
        self,
        features: FloatArray,
        prices: FloatArray,
        window: int = 20,
        transaction_cost: float = 0.001,
        eta: float = 0.01,
    ) -> None:
        """
        Parameters
        ----------
        features:
            Normalised feature array of shape (T, 15) — the output of
            data.load_data()["train"]. Columns are ordered as described in
            data.py: [asset0_logret, asset0_vol, asset0_meanrev, ...].
        prices:
            Raw adjusted close prices of shape (T, 5). Used to recompute
            log returns inside step() to cross-check against the feature
            matrix (currently only prices is stored for future use; log
            returns are read directly from `features`).
        window:
            Look-back window (in trading days) for the state's return
            history component.
        transaction_cost:
            Proportional cost applied to the L1 change in weights at each
            rebalance step. A value of 0.001 (10 bps) is a realistic proxy
            for ETF bid-ask spreads.
        eta:
            EMA decay rate for the differential Sharpe accumulators A and B.
            Smaller eta gives a longer-memory estimate; larger eta responds
            faster to recent returns. Typical values: 0.01–0.05.
        """
        super().__init__()

        self._features = features
        self._prices = prices
        self._window = window
        self._transaction_cost = transaction_cost
        self._eta = eta

        self._T: int = features.shape[0]

        # Observation size: window log-return history per asset,
        # plus vol + meanrev + weights per asset, plus portfolio vol scalar.
        obs_size: int = (
            window * self.N_ASSETS + self.N_ASSETS + self.N_ASSETS + self.N_ASSETS + 1
        )

        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        # The agent emits raw logits; softmax normalises them inside step().
        self.action_space: spaces.Box = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.N_ASSETS,),
            dtype=np.float32,
        )

        # State variables — initialised properly in reset()
        self._t: int = 0
        self._weights: FloatArray = np.zeros(self.N_ASSETS, dtype=np.float32)
        self._A: float = 0.0  # EMA of returns
        self._B: float = 0.0  # EMA of squared returns

    # ------------------------------------------------------------------
    # Public Gymnasium interface
    # ------------------------------------------------------------------

    @override
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> ResetReturn:
        """
        Initialise a new episode.

        - Portfolio weights start at equal weight (1/N each).
        - EMA accumulators A and B start at zero.
        - Current timestep is sampled uniformly from [window, T-1] so that
          the look-back window is always fully populated and the episode has
          at least one step remaining before termination.

        Returns
        -------
        obs:
            Initial observation vector, shape (obs_size,).
        info:
            Empty dict (no extra info at reset time).
        """
        super().reset(seed=seed)

        # Equal-weight initialisation
        self._weights = np.full(self.N_ASSETS, 1.0 / self.N_ASSETS, dtype=np.float32)

        # EMA accumulators: pre-warm from window steps of prior history so
        # the differential Sharpe denominator is non-degenerate from step 1.
        self._A = 0.0
        self._B = 0.0

        # Sample a random starting timestep. We need at least `window` prior
        # rows for the look-back window and at least one step before T-1.
        self._t = int(self.np_random.integers(self._window, self._T - 1))

        logret_cols: list[int] = [
            i * self.N_FEATURES_PER_ASSET + self.LOGRET_OFFSET
            for i in range(self.N_ASSETS)
        ]
        for s in range(self._t - self._window, self._t):
            R: float = float(np.dot(self._weights, self._features[s, logret_cols]))
            self._A += self._eta * (R - self._A)
            self._B += self._eta * (R * R - self._B)

        obs = self._get_obs()
        return obs, {}

    @override
    def step(self, action: FloatArray) -> StepReturn:
        """
        Apply one rebalancing step.

        The action (raw logits) is softmax-normalised to a valid weight
        vector, then the portfolio return and differential Sharpe reward
        are computed.

        Parameters
        ----------
        action:
            Raw logit vector of shape (N_ASSETS,) with values in [0, 1].

        Returns
        -------
        obs:
            Next observation after advancing the timestep.
        reward:
            Differential Sharpe ratio reward net of transaction costs.
        terminated:
            True when the episode reaches the last available timestep.
        truncated:
            Always False — no time-limit truncation.
        info:
            Dict containing ``portfolio_return``, ``transaction_cost``,
            and ``weights`` (the softmax-normalised weight vector applied
            at this step).
        """
        # ------------------------------------------------------------------
        # 1. Normalise action to portfolio weights via softmax
        # ------------------------------------------------------------------
        new_weights = self._softmax(action)

        # Sanity check: softmax output must sum to 1.0 within floating-point
        # tolerance. If this fails, something is wrong with _softmax().
        assert (
            abs(new_weights.sum() - 1.0) < 1e-5
        ), f"Softmax weights do not sum to 1: sum={new_weights.sum():.8f}"

        # ------------------------------------------------------------------
        # 2. Compute portfolio return at the current timestep
        #
        # Log returns are read from the normalised feature matrix. The
        # z-score scale acts as an implicit variance stabiliser across
        # different market regimes (H1 tested raw price returns — val
        # Sharpe dropped from 0.52 to 0.28, so we keep normalised returns).
        #
        # The agent sets weights at _t and earns the return realised
        # between _t and _t+1, so we read features[_t+1].
        # ------------------------------------------------------------------
        logret_cols: list[int] = [
            i * self.N_FEATURES_PER_ASSET + self.LOGRET_OFFSET
            for i in range(self.N_ASSETS)
        ]
        asset_returns: FloatArray = self._features[
            self._t + 1, logret_cols
        ]  # shape (N,)

        # Portfolio return: weighted sum of individual asset log returns
        portfolio_return: float = float(np.dot(new_weights, asset_returns))

        # ------------------------------------------------------------------
        # 3. Compute transaction cost
        #
        # Proportional to the L1 norm of the weight change. This penalises
        # high-turnover policies that would be expensive to execute in
        # practice.
        # ------------------------------------------------------------------
        weight_change: float = float(np.sum(np.abs(new_weights - self._weights)))
        cost: float = self._transaction_cost * weight_change

        # ------------------------------------------------------------------
        # 4. Update EMA accumulators for the differential Sharpe reward
        #
        # We maintain two exponential moving averages:
        #   A_t  — EMA of raw returns  (tracks the mean return)
        #   B_t  — EMA of squared returns  (tracks the mean squared return)
        #
        # The update rule is:
        #   A_t = A_{t-1} + eta * (R_t - A_{t-1})
        #   B_t = B_{t-1} + eta * (R_t^2 - B_{t-1})
        #
        # This is a standard EMA with decay factor eta. Storing A and B
        # (rather than variance directly) matches the Moody & Saffell (1998)
        # formulation and makes the gradient derivation straightforward.
        # ------------------------------------------------------------------
        R: float = portfolio_return
        R_sq: float = R * R

        # Save previous values before update — they appear in the numerator
        # of the differential Sharpe formula
        A_prev: float = self._A
        B_prev: float = self._B

        delta_A: float = R - A_prev
        delta_B: float = R_sq - B_prev

        self._A = A_prev + self._eta * delta_A
        self._B = B_prev + self._eta * delta_B

        # ------------------------------------------------------------------
        # 5. Compute differential Sharpe ratio
        #
        # The differential Sharpe D_t is the instantaneous gradient of the
        # Sharpe ratio S_t = A_t / sqrt(B_t - A_t^2) with respect to the
        # EMA smoothing parameter eta. It equals:
        #
        #   D_t = (B_{t-1} * delta_A_t - 0.5 * A_{t-1} * delta_B_t)
        #         / (B_{t-1} - A_{t-1}^2)^(3/2)
        #
        # Intuitively: numerator rewards new returns that increase the mean
        # (B_prev * delta_A > 0) while penalising returns that inflate
        # variance (A_prev * delta_B term). The denominator is the current
        # variance estimate raised to the 3/2 power.
        #
        # We add SHARPE_EPS before the power to guard against zero variance
        # early in training (when A and B are still near zero).
        # ------------------------------------------------------------------
        variance_est: float = B_prev - A_prev * A_prev
        # Clamp to zero before taking the power to avoid negative radicand
        # due to floating-point rounding (variance must be non-negative)
        variance_est = max(variance_est, 0.0)
        denom: float = (variance_est + self.SHARPE_EPS) ** 1.5

        numerator: float = B_prev * delta_A - 0.5 * A_prev * delta_B
        differential_sharpe: float = numerator / denom

        # ------------------------------------------------------------------
        # 6. Net reward = differential Sharpe - transaction cost
        # ------------------------------------------------------------------
        reward: float = differential_sharpe - cost

        assert not np.isnan(
            reward
        ), f"NaN reward at t={self._t}: DS={differential_sharpe:.6f}, cost={cost:.6f}"

        # ------------------------------------------------------------------
        # 7. Advance state
        # ------------------------------------------------------------------
        self._weights = new_weights
        self._t += 1

        # Terminate when _t == T-2 (after increment) because the next step would
        # try to read features[_t + 1] = features[T-1], which is the last valid
        # row. We must not advance past it.
        terminated: bool = self._t >= self._T - 2

        obs = self._get_obs()

        info: dict[str, Any] = {
            "portfolio_return": portfolio_return,
            "transaction_cost": cost,
            "weights": new_weights,
        }

        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> FloatArray:
        """
        Build and return the current observation vector (shape: obs_size,).

        Components (concatenated in this order):
          1. Log returns for each asset over the past `window` days,
             flattened in time-major order: [t-window, t-window+1, ..., t-1]
             for each asset. Shape: (window * N_ASSETS,).
          2. Rolling volatility of each asset at timestep t. Shape: (N_ASSETS,).
          3. Mean-reversion signal of each asset at timestep t. Shape: (N_ASSETS,).
          4. Current portfolio weights. Shape: (N_ASSETS,).
          5. Current portfolio volatility (std of weighted returns over the
             look-back window). Shape: (1,).
        """
        t = self._t

        # ------------------------------------------------------------------
        # Component 1: per-asset log returns over the look-back window
        #
        # Slice rows [t - window, t) from the feature matrix and extract
        # the logret column for each asset. The result is (window, N_ASSETS).
        # We flatten it row-major so the agent sees returns in time order.
        # ------------------------------------------------------------------
        logret_cols: list[int] = [
            i * self.N_FEATURES_PER_ASSET + self.LOGRET_OFFSET
            for i in range(self.N_ASSETS)
        ]
        window_logrets: FloatArray = self._features[
            t - self._window : t, logret_cols
        ]  # shape (window, N_ASSETS)
        logret_flat: FloatArray = window_logrets.flatten().astype(np.float32)

        # ------------------------------------------------------------------
        # Component 2: rolling volatility at the current timestep
        # ------------------------------------------------------------------
        vol_cols: list[int] = [
            i * self.N_FEATURES_PER_ASSET + self.VOL_OFFSET
            for i in range(self.N_ASSETS)
        ]
        vol: FloatArray = self._features[t, vol_cols].astype(np.float32)

        # ------------------------------------------------------------------
        # Component 3: mean-reversion signal at the current timestep
        # ------------------------------------------------------------------
        meanrev_cols: list[int] = [
            i * self.N_FEATURES_PER_ASSET + self.MEANREV_OFFSET
            for i in range(self.N_ASSETS)
        ]
        meanrev: FloatArray = self._features[t, meanrev_cols].astype(np.float32)

        # ------------------------------------------------------------------
        # Component 4: current portfolio weights
        # ------------------------------------------------------------------
        weights: FloatArray = self._weights.astype(np.float32)

        # ------------------------------------------------------------------
        # Component 5: current portfolio volatility
        #
        # Computed as the standard deviation of the portfolio's daily returns
        # over the look-back window. This gives the agent a scalar measure of
        # how risky the current allocation has been recently.
        # ------------------------------------------------------------------
        portfolio_window_returns: FloatArray = (window_logrets @ self._weights).astype(
            np.float32
        )  # shape (window,)
        # Use ddof=1 (sample std) to match the rolling std used in data.py,
        # so both vol estimates in the observation are on the same scale.
        portfolio_vol: FloatArray = np.array(
            [portfolio_window_returns.std(ddof=1)], dtype=np.float32
        )  # shape (1,)

        obs: FloatArray = np.concatenate(
            [logret_flat, vol, meanrev, weights, portfolio_vol]
        )

        assert not np.isnan(obs).any(), (
            f"NaN detected in observation at t={t}: "
            f"logret_nan={np.isnan(logret_flat).any()}, "
            f"vol_nan={np.isnan(vol).any()}, "
            f"meanrev_nan={np.isnan(meanrev).any()}, "
            f"weights_nan={np.isnan(weights).any()}, "
            f"port_vol_nan={np.isnan(portfolio_vol).any()}"
        )

        return obs

    @staticmethod
    def _softmax(x: FloatArray) -> FloatArray:
        """
        Numerically stable softmax normalisation.

        Subtracting the maximum before exponentiating prevents overflow for
        large logit values without changing the output (the shift cancels in
        the ratio). The result always sums to exactly 1.0 in float32
        arithmetic (up to rounding).

        Parameters
        ----------
        x:
            Raw logit vector of any shape.

        Returns
        -------
        Softmax-normalised vector of the same shape and dtype float32.
        """
        x = x.astype(np.float64)  # use float64 for the intermediate computation
        x_shifted = x - x.max()
        exp_x = np.exp(x_shifted)
        return (exp_x / exp_x.sum()).astype(np.float32)
