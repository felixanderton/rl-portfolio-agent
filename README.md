# rl-portfolio-agent

A reinforcement learning agent that allocates capital across five US sector ETFs using Proximal Policy Optimisation (PPO). The agent is trained with a Differential Sharpe Ratio reward signal — a dense, differentiable approximation of the Sharpe ratio that provides a risk-adjusted gradient at every timestep rather than waiting until the end of an episode.

---

## Table of contents

- [MDP Formulation](#mdp-formulation)
- [Why Differential Sharpe?](#why-differential-sharpe)
- [Limitations and Failure Modes](#limitations-and-failure-modes)
- [File Structure](#file-structure)
- [Installation and Usage](#installation-and-usage)

---

## MDP Formulation

The portfolio allocation problem is cast as a finite-horizon Markov Decision Process `(S, A, P, R)`.

### State space

The observation vector has **116 dimensions**, constructed as follows at each timestep `t`:

| Component | Size | Description |
|---|---|---|
| Log-return history | 20 x 5 = 100 | Z-score normalised log returns for each of the 5 assets over the past 20 trading days, in time-major order |
| Rolling volatility | 5 | 20-day rolling standard deviation of log returns for each asset at time `t`, z-score normalised |
| Mean-reversion signal | 5 | Z-score of each asset's price relative to its 20-day rolling mean and std: `(P_t - mu) / sigma`, z-score normalised |
| Portfolio weights | 5 | Current allocation across the 5 assets |
| Portfolio volatility | 1 | Standard deviation of the portfolio's daily returns over the 20-day look-back window |

**Why each component:**

- **Log-return history** gives the agent a direct view of recent price dynamics. Log returns are used because they are additive over time and approximately normally distributed for small moves.
- **Rolling volatility** captures each asset's current risk level so the agent can shift away from high-volatility names during stressed periods.
- **Mean-reversion signal** provides a stationary, scaled measure of where each asset sits relative to its recent trend, encoding potential mean-reversion opportunities without requiring the agent to learn a normalisation itself.
- **Portfolio weights** close the feedback loop: the agent must know its current allocation to reason about the cost and direction of any rebalancing.
- **Portfolio volatility** gives a scalar summary of how much risk the current allocation has been realising, separate from the per-asset volatility signals.

### Action space

```
Box(low=0.0, high=1.0, shape=(5,), dtype=float32)
```

The agent emits a vector of raw logits, one per asset. The environment applies a **softmax transformation** inside `step()` before using them as portfolio weights.

Softmax is preferred over direct weight output for two reasons. First, it guarantees that the weights are non-negative and sum to exactly one at every step, without requiring the agent to learn this constraint through reward shaping. Second, it keeps the action space unconstrained from the policy's perspective — the agent can express any valid weight vector by adjusting the relative magnitudes of its logits, and the gradient of the softmax is well-defined everywhere, which aids stable learning with PPO.

### Transition function

The environment is **deterministic given the data**. Market dynamics are exogenous: at each step the environment simply advances the timestep by one and reads the next row of pre-computed, normalised features from the dataset. There is no stochastic model of price dynamics; the agent's actions influence its portfolio value and EMA accumulators but do not alter the underlying price series. The only source of randomness is the episode start index, which is sampled uniformly from `[window, T-1]` at the beginning of each episode.

### Reward function

The reward at each step is the **Differential Sharpe Ratio** (Moody & Saffell, 1998) minus a proportional transaction cost:

```
reward_t = D_t - c * ||w_t - w_{t-1}||_1
```

where `c = 0.001` (10 basis points) and `w_t` are the softmax-normalised weights applied at step `t`.

**Derivation of the Differential Sharpe.** The standard Sharpe ratio at time `t` is:

```
S_t = A_t / sqrt(B_t - A_t^2)
```

where `A_t` and `B_t` are exponential moving averages of the portfolio return `R_t` and its square:

```
A_t = A_{t-1} + eta * (R_t - A_{t-1})      [EMA of returns]
B_t = B_{t-1} + eta * (R_t^2 - B_{t-1})    [EMA of squared returns]
```

with decay parameter `eta` (default `0.01`). The quantity `B_t - A_t^2` is the EMA estimate of the variance of `R`.

The **Differential Sharpe** `D_t` is defined as the gradient of `S_t` with respect to `eta`, evaluated at infinitesimal `eta`:

```
D_t = dS_t / d(eta)
    = (B_{t-1} * delta_A_t  -  0.5 * A_{t-1} * delta_B_t)
      / (B_{t-1} - A_{t-1}^2)^(3/2)
```

where:

```
delta_A_t = R_t - A_{t-1}
delta_B_t = R_t^2 - B_{t-1}
```

**Intuition.** The numerator rewards steps where the new return increases the estimated mean (the `B_{t-1} * delta_A_t` term) while penalising steps where the new return inflates the estimated variance (the `0.5 * A_{t-1} * delta_B_t` term). The denominator normalises by the current variance estimate raised to the 3/2 power, so the signal is scale-invariant. A small epsilon guard `(1e-8)` is added to the denominator before the power to prevent division by zero when `A` and `B` are near zero early in an episode.

---

## Why Differential Sharpe?

The choice of reward function has a direct effect on what the agent learns to optimise. The alternatives are:

**Raw portfolio return.** Rewards each step's portfolio return directly. The problem is that this gives no penalty for variance: an agent maximising raw return will concentrate the portfolio in the highest-expected-return asset regardless of the associated risk, producing a policy that is volatile and difficult to execute in practice.

**End-of-episode Sharpe.** Compute the full Sharpe ratio of the episode's return series and assign it as a terminal reward. This is sparse: the agent receives a gradient signal only at the final step of each episode, which makes credit assignment across hundreds of timesteps unreliable and training unstable. It also means the agent receives no feedback during the episode about whether individual allocation decisions are contributing to or detracting from the risk-adjusted objective.

**Risk-adjusted return (e.g. Sortino).** Penalises downside variance only. While theoretically appealing, estimating the downside deviation reliably requires a longer history than a typical episode provides. The Sortino denominator is more volatile and harder to differentiate through, and the signal it produces at each step is noisier than the Differential Sharpe for the same episode length.

**Differential Sharpe (chosen).** Provides a dense, differentiable approximation to the Sharpe ratio at every timestep. Because the EMA accumulators `A` and `B` carry information forward across the episode, the agent's reward at step `t` implicitly reflects the entire history of returns since the episode began. This combines the interpretability of a risk-adjusted objective with the stability benefits of a dense reward signal, and it avoids the credit assignment problem of sparse terminal rewards.

---

## Limitations and Failure Modes

### Overfitting to the training regime

The feature normalisation statistics (mean and standard deviation per feature column) and the hyperparameter grid are computed on and tuned to the 2000-2015 training period. This period includes the 2008 financial crisis but not the low-volatility, low-interest-rate environment of 2016-2019 or the COVID-driven volatility of 2020. A regime change that shifts the distribution of returns, volatilities, or cross-asset correlations outside the range seen during training may cause the agent's learned policy to produce poor allocations even if the policy was well-calibrated on the training set.

### EMA warm-up

At the start of each episode, the EMA accumulators `A` and `B` are reset to zero. Because the EMA has not yet accumulated enough history, the variance estimate `B - A^2` is very small and the denominator of the Differential Sharpe is dominated by the epsilon guard `(1e-8)`. As a result, rewards in approximately the first 50 steps of each episode do not accurately reflect the Sharpe ratio the EMA is converging toward. The policy learns correct behaviour for later steps in an episode but may behave inconsistently near episode starts.

### Z-score normalised returns in reward

The reward is computed using the z-score normalised log returns stored in the feature array, not the raw log returns. The EMA accumulators `A` and `B` therefore operate in standardised units rather than in true return units. The resulting Differential Sharpe value is not directly comparable to a Sharpe ratio computed from raw returns, and its magnitude is not interpretable as a standard annualised Sharpe. It is useful as a relative training signal but should not be read as an absolute risk-adjusted performance number.

### Lookback window of 20 days

The observation vector contains only 20 days of return history. The agent cannot directly observe trends or regime shifts that unfold over months or quarters. A structural change — for example, a sustained rise in cross-sector correlation during a market stress event — will only influence the agent's behaviour once it propagates into the 20-day window. Long-duration mean reversion, momentum, or macro regime signals are not representable in the current state space.

### Transaction cost model

The 0.001 (10 bps) per-unit flat cost is a simple first-order proxy for trading friction. Real execution costs for ETFs involve bid-ask spreads, market impact, and borrow costs, all of which grow non-linearly with trade size. A large rebalancing trade in an illiquid market can move the price against the trader, meaning the actual cost is higher than the flat-rate model predicts. The current reward signal therefore understates the true cost of high-turnover policies.

### Five-asset universe

The portfolio is restricted to five US sector ETFs (XLK, XLE, XLF, XLV, XLI). These are all domestic equity sectors and are therefore highly correlated during systemic risk events such as the 2020 COVID crash or the 2008 financial crisis. In those episodes, all five assets fell simultaneously, leaving the agent with no diversification-based escape: any allocation across the five will realise similar drawdowns. The agent cannot allocate to bonds, gold, international equities, or cash, which limits its ability to manage tail risk.

---

## File Structure

| File | Description |
|---|---|
| `data.py` | Downloads adjusted close prices via yfinance, engineers features (log return, rolling volatility, mean-reversion signal), normalises using training statistics only, and returns train/val/test splits |
| `environment.py` | Custom Gymnasium environment implementing the MDP: observation construction, softmax action normalisation, Differential Sharpe reward computation, and transaction cost penalty |
| `baselines.py` | Three benchmark strategies (equal weight, cross-sectional momentum, buy-and-hold SPY) and a shared `evaluate_portfolio` function computing annualised return, volatility, Sharpe, max drawdown, and Calmar ratio |
| `train.py` | PPO training loop with grid search over learning rate, rollout length, and entropy coefficient; TensorBoard logging; checkpoint saving; best-model selection by validation Sharpe |
| `evaluate.py` | Loads the saved best model and runs it deterministically on the test split; compares metrics against the three baselines |
| `ablation.py` | Ablation study runner for evaluating the contribution of individual observation components and reward choices |

---

## Installation and Usage

**Requirements:** Python 3.12+

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train

```bash
python train.py
```

This runs a grid search over 4 hyperparameter combinations (2 learning rates x 2 rollout lengths). Each combination trains for 500,000 environment steps. The best model by validation Sharpe ratio is saved to `best_model/best_model.zip`. TensorBoard logs are written to `logs/` and periodic checkpoints to `checkpoints/`.

To monitor training in real time:

```bash
tensorboard --logdir logs/
```

### Evaluate

```bash
python evaluate.py
```

Loads `best_model/best_model.zip` and runs a deterministic episode over the test split (2019-2025). Prints a metrics table comparing the RL agent against the equal-weight, momentum, and buy-and-hold SPY baselines.

### Data split

| Split | Period | Purpose |
|---|---|---|
| Train | 2000-01-01 to 2015-01-01 | Policy learning |
| Validation | 2015-01-01 to 2019-01-01 | Hyperparameter selection |
| Test | 2019-01-01 to 2025-01-01 | Final evaluation (held out) |

Normalisation statistics are computed on the training split only and applied to all three splits to prevent data leakage.
