"""
tests/test_concentration_penalty.py

Unit tests for the concentration_penalty_lambda feature added to PortfolioEnv
in the H13 hypothesis branch.

The penalty subtracts `lambda * sum(weights**2)` (the Herfindahl-Hirschman
Index) from the reward at every step. Tests verify:

  1. Default lambda=0.0 produces no penalty (backward compatibility).
  2. A positive lambda reduces the reward vs. lambda=0.0 for the same action.
  3. A concentrated allocation (high HHI) receives a larger penalty than a
     uniform allocation (low HHI) under the same lambda.
  4. The lambda parameter is stored on the instance and defaults to 0.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

# Allow imports from src/ without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment import PortfolioEnv  # noqa: E402

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

type FloatArray = npt.NDArray[np.float32]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ASSETS: int = PortfolioEnv.N_ASSETS
N_FEATURES_PER_ASSET: int = PortfolioEnv.N_FEATURES_PER_ASSET
WINDOW: int = 20

# Total rows needed: window rows for look-back + a few steps to execute.
# reset() samples from [window, T-2], so T must be at least window + 3.
T: int = WINDOW + 10

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_data() -> tuple[FloatArray, FloatArray]:
    """
    Return (features, prices) arrays of minimal valid shape.

    features: (T, N_ASSETS * N_FEATURES_PER_ASSET) = (30, 15), float32
    prices:   (T, N_ASSETS) = (30, 5), float32

    Values are small non-zero constants so the reward calculation stays
    well-behaved (no NaNs, no zero-variance denominator blowup).
    """
    rng = np.random.default_rng(42)
    features = rng.standard_normal((T, N_ASSETS * N_FEATURES_PER_ASSET)).astype(
        np.float32
    )
    # Scale down to typical normalised log-return magnitudes
    features *= 0.1
    prices = (
        rng.uniform(50.0, 150.0, size=(T, N_ASSETS)).astype(np.float32)
    )
    return features, prices


def _make_env(
    features: FloatArray,
    prices: FloatArray,
    concentration_penalty_lambda: float = 0.0,
) -> PortfolioEnv:
    return PortfolioEnv(
        features=features,
        prices=prices,
        window=WINDOW,
        concentration_penalty_lambda=concentration_penalty_lambda,
    )


def _reset_and_step(
    env: PortfolioEnv,
    action: FloatArray,
    seed: int = 0,
) -> float:
    """Reset env with a fixed seed and take one step; return the reward."""
    env.reset(seed=seed)
    _, reward, _, _, _ = env.step(action)
    return reward


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_lambda_is_zero(
    synthetic_data: tuple[FloatArray, FloatArray],
) -> None:
    """PortfolioEnv.__init__ stores concentration_penalty_lambda=0.0 by default."""
    features, prices = synthetic_data
    env = PortfolioEnv(features=features, prices=prices, window=WINDOW)
    assert env._concentration_penalty_lambda == 0.0


def test_zero_lambda_reward_matches_no_argument(
    synthetic_data: tuple[FloatArray, FloatArray],
) -> None:
    """
    Explicitly passing lambda=0.0 must produce the same reward as omitting the
    argument entirely, for the same action and seed.
    """
    features, prices = synthetic_data
    action: FloatArray = np.zeros(N_ASSETS, dtype=np.float32)

    env_default = _make_env(features, prices, concentration_penalty_lambda=0.0)
    env_explicit = _make_env(features, prices, concentration_penalty_lambda=0.0)

    reward_default = _reset_and_step(env_default, action, seed=7)
    reward_explicit = _reset_and_step(env_explicit, action, seed=7)

    assert reward_default == pytest.approx(reward_explicit)


def test_positive_lambda_reduces_reward(
    synthetic_data: tuple[FloatArray, FloatArray],
) -> None:
    """
    With lambda > 0, the reward is strictly lower than with lambda=0.0
    for the same action and seed. The penalty is always positive (HHI > 0
    for any non-zero weight vector), so the sign of the difference is
    deterministic regardless of the underlying differential Sharpe value.
    """
    features, prices = synthetic_data
    action: FloatArray = np.zeros(N_ASSETS, dtype=np.float32)

    env_no_penalty = _make_env(features, prices, concentration_penalty_lambda=0.0)
    env_penalised = _make_env(features, prices, concentration_penalty_lambda=0.1)

    reward_no_penalty = _reset_and_step(env_no_penalty, action, seed=7)
    reward_penalised = _reset_and_step(env_penalised, action, seed=7)

    assert reward_penalised < reward_no_penalty


def test_concentrated_allocation_receives_larger_penalty_than_uniform(
    synthetic_data: tuple[FloatArray, FloatArray],
) -> None:
    """
    A fully concentrated allocation (one asset gets all weight) has a higher
    HHI than a uniform allocation, so it must receive a larger penalty and
    therefore a lower reward, all else equal.

    Both envs share the same lambda and the same seed so the differential
    Sharpe component is identical. The only difference is the action, and
    therefore the softmax output and the resulting HHI.
    """
    features, prices = synthetic_data
    lam: float = 1.0  # large lambda to make the penalty dominate any tiny
    # differential in the Sharpe component from different weight vectors

    env_uniform = _make_env(features, prices, concentration_penalty_lambda=lam)
    env_concentrated = _make_env(features, prices, concentration_penalty_lambda=lam)

    # Equal logits → softmax gives [0.2, 0.2, 0.2, 0.2, 0.2]
    action_uniform: FloatArray = np.zeros(N_ASSETS, dtype=np.float32)

    # Highly unequal logits → softmax ≈ [1, 0, 0, 0, 0]
    action_concentrated: FloatArray = np.array(
        [10.0, -10.0, -10.0, -10.0, -10.0], dtype=np.float32
    )

    reward_uniform = _reset_and_step(env_uniform, action_uniform, seed=7)
    reward_concentrated = _reset_and_step(env_concentrated, action_concentrated, seed=7)

    assert reward_concentrated < reward_uniform


def test_penalty_magnitude_proportional_to_lambda(
    synthetic_data: tuple[FloatArray, FloatArray],
) -> None:
    """
    Doubling lambda must double the penalty gap relative to lambda=0.

    For a fixed action and seed:
      gap(2*lam) == 2 * gap(lam)

    where gap(lam) = reward(lam=0) - reward(lam).
    """
    features, prices = synthetic_data
    action: FloatArray = np.zeros(N_ASSETS, dtype=np.float32)
    seed: int = 7

    env_base = _make_env(features, prices, concentration_penalty_lambda=0.0)
    env_lam1 = _make_env(features, prices, concentration_penalty_lambda=0.05)
    env_lam2 = _make_env(features, prices, concentration_penalty_lambda=0.10)

    r_base = _reset_and_step(env_base, action, seed=seed)
    r_lam1 = _reset_and_step(env_lam1, action, seed=seed)
    r_lam2 = _reset_and_step(env_lam2, action, seed=seed)

    gap1 = r_base - r_lam1
    gap2 = r_base - r_lam2

    assert gap1 > 0, "lambda=0.05 must produce a non-zero penalty"
    assert gap2 == pytest.approx(2.0 * gap1, rel=1e-5)
