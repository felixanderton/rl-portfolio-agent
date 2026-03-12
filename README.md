# rl-portfolio-agent

A reinforcement learning agent for dynamic sector ETF allocation, built as a systematic ML research project. The agent uses Proximal Policy Optimisation (PPO) to allocate capital across five US sector ETFs, trained with a Differential Sharpe Ratio reward and evaluated against equal-weight, momentum, and buy-and-hold SPY baselines.

The primary goal was to learn rigorous ML workflows on time-series data: proper experiment tracking, train/val/test discipline, hypothesis-driven iteration, and honest evaluation. Fourteen hypotheses were tested over the course of the project.

---

## Table of Contents

- [Results](#results)
- [MDP Formulation](#mdp-formulation)
- [Experimental Journey](#experimental-journey)
- [Infrastructure](#infrastructure)
- [File Structure](#file-structure)
- [Installation and Usage](#installation-and-usage)

---

## Results

| Model | Val Sharpe | Test Sharpe | Notes |
|---|---|---|---|
| Equal weight baseline | ~0.30 | 0.648 | Benchmark |
| Momentum baseline | — | 0.727 | 12-month cross-sectional momentum |
| SPY buy & hold | — | 0.800 | Passive benchmark |
| H10 (TC curriculum + extended training) | 0.813 (peak) | **0.855** | Selected checkpoint; 95% CI [−0.23, +0.61] vs equal-weight |

**Interpretation:** The H10 agent outperforms all baselines on the held-out 2019–2024 test period (Sharpe 0.855 vs 0.800 for SPY buy-and-hold), with higher annualised returns (+20.9%) and lower max drawdown than equal-weight. The bootstrap confidence interval on the Sharpe difference vs equal-weight is wide and crosses zero, reflecting the limited statistical power of a single 6-year evaluation window — the result is positive but not conclusive. The val Sharpe of 0.81 was achieved with warm-starting across multiple runs; a clean from-scratch run (H13) produced val Sharpe of 0.46, indicating the high val/test numbers depend on accumulated training budget rather than a single training run.

---

## MDP Formulation

The portfolio allocation problem is cast as a finite-horizon Markov Decision Process `(S, A, P, R)`.

### State space

The observation vector has **116 dimensions**, constructed as follows at each timestep `t`:

| Component | Size | Description |
|---|---|---|
| Log-return history | 20 × 5 = 100 | Z-score normalised log returns for each asset over the past 20 trading days, in time-major order |
| Rolling volatility | 5 | 20-day rolling standard deviation of log returns, z-score normalised |
| Mean-reversion signal | 5 | Z-score of price relative to its 20-day rolling mean: `(P_t - mu) / sigma` |
| Portfolio weights | 5 | Current allocation across the 5 assets |
| Portfolio volatility | 1 | Standard deviation of portfolio daily returns over the 20-day look-back |

### Action space

```
Box(low=0.0, high=1.0, shape=(5,), dtype=float32)
```

The agent emits raw logits which the environment converts to portfolio weights via softmax, guaranteeing non-negativity and sum-to-one at every step without requiring the agent to learn this constraint.

### Reward function

```
reward_t = DifferentialSharpe(R_t) - c * ||w_t - w_{t-1}||_1
```

where `c` ramps from 0.0002 to 0.001 quadratically over the training run (transaction cost curriculum, H6).

**Differential Sharpe Ratio** (Moody & Saffell, 1998) provides a dense, differentiable approximation to the Sharpe ratio at every timestep via EMA accumulators `A_t` (mean return) and `B_t` (mean squared return):

```
D_t = (B_{t-1} * delta_A_t  -  0.5 * A_{t-1} * delta_B_t) / (B_{t-1} - A_{t-1}^2)^(3/2)
```

This combines the interpretability of a risk-adjusted objective with the stability of dense rewards — the agent gets a gradient signal at every step, not just at episode end.

**Transaction cost curriculum** (H6): ramping `c` from 0.0002 to 0.001 quadratically over the training run allows free early exploration before progressively penalising turnover. This was the only regularisation approach that consistently improved val Sharpe.

---

## Experimental Journey

Fourteen hypotheses were tested systematically. The key findings:

### The overfitting problem

By H4, the agent was achieving training Sharpe ratios of 4–7 while val Sharpe stagnated around 0.5–0.7. This train/val gap — larger than any other project I've seen documented — became the central problem.

The policy was learning to exploit specific trajectories in the training data: taking concentrated positions that happened to be correct for memorised patterns rather than learning generalisable allocation rules. This manifested as extremely high entropy loss collapse (the policy converging to near-deterministic behaviour), high turnover (1.0–1.5 daily L1 weight change), and val Sharpe degrading monotonically after ~450k steps despite training Sharpe continuing to rise.

### What regularisation approaches were tried

| Hypothesis | Approach | Outcome |
|---|---|---|
| H5 | Weight decay (L2 on policy network) | Val Sharpe −18% — interfered with gradient dynamics |
| H6 | TC curriculum (ramp transaction cost) | +9.5% — the only successful regulariser |
| H7 | Block bootstrap episode augmentation | Training destabilised, early termination |
| H12 | Observation noise (sigma=0.05) | Flat val Sharpe, no improvement |
| H13 | Portfolio concentration penalty (HHI) | Val Sharpe −35% — suppressed skill alongside memorisation |

The pattern: regularisers that operate in the reward space (H6) can safely constrain behaviour without disrupting policy gradient dynamics. Regularisers that operate in weight space (H5) or input space (H12) tend to interfere with the gradient updates that produce the late-training val Sharpe surge.

The concentration penalty (H13) failed for a specific reason: concentrated positions are *both* the mechanism of overfitting *and* the mechanism of genuine skill in a 5-asset universe. A penalty that discourages concentration cannot distinguish between the two.

### The feature ceiling

The more important finding is that the features themselves may be the binding constraint. The original feature set (20-day log returns, volatility, mean-reversion) operates entirely within a 20-day lookback. The primary documented driver of sector ETF performance — cross-sectional price momentum at 3–12 month horizons — is invisible to this feature set.

The training period (2000–2014) and validation period (2015–2018) also exhibit regime differences that price features cannot bridge: the Trump rally (2016–17) and the beginning of the rate-hiking cycle (2017–18) were driven by expectations of sector-specific policy changes and macroeconomic shifts, none of which appear in 20-day price windows.

H14 adds 63-day and 252-day cumulative log returns to directly test whether medium/long-horizon momentum improves the feature ceiling. It is the only hypothesis that addresses the root cause rather than the symptoms.

### What the numbers actually mean

A val Sharpe above ~0.30 (roughly equal-weight) with a statistically significant bootstrap confidence interval on the test set would be a meaningful result. The H6 number of 0.71 is interesting but not trustworthy without a clean test-set evaluation. If H14 achieves val Sharpe of 0.5+ with a positive test-set result, that would be a genuine finding attributable to the momentum features.

---

## Infrastructure

The project uses production-grade ML infrastructure throughout:

- **ClearML** for experiment tracking — all hyperparameters, metrics, plots, and model artifacts are logged automatically. The full history of 14 hypothesis runs is available in the project dashboard.
- **Modal** for cloud training — each hypothesis runs on a 16-core cloud instance with 32 GB RAM. Checkpoints are persisted to a Modal Volume so preempted runs auto-resume from the last checkpoint.
- **Systematic hypothesis testing** — each change is a named hypothesis with a written prediction, falsification criterion, and result logged to `docs/HYPOTHESES.md` and `docs/EXPERIMENT_LOG.md`.
- **Proper data splits** — train (2000–2015), val (2015–2019), test (2019–2025). Normalisation statistics computed on train only. Test set was held out until automatic test evaluation was added to the pipeline in H13; it has not been used for any model selection decision.
- **Bootstrap CI** — `evaluate.py` computes a 95% bootstrap confidence interval on the Sharpe difference between the PPO agent and equal-weight baseline, enabling statistical claims rather than point estimates.

---

## File Structure

```
src/
  data.py           Downloads prices, engineers features (log return, vol, mean-reversion,
                    63/252-day momentum), normalises on train stats, returns splits
  environment.py    Gymnasium env: observation construction, softmax actions,
                    differential Sharpe reward, TC penalty, concentration penalty
  train.py          PPO training loop with TC curriculum, periodic val evaluation,
                    ClearML logging, checkpoint saving, test-set evaluation on completion
  evaluate.py       Full test-set evaluation: metrics table, bootstrap CI, equity curves,
                    rolling Sharpe, weight heatmap
  baselines.py      Equal-weight, momentum, buy-and-hold SPY strategies + evaluate_portfolio
  modal_train.py    Modal cloud training: Volume checkpoint persistence, SIGTERM handler,
                    auto-resume on preemption, warm-start from ClearML artifacts
  ablation.py       Ablation study runner

docs/
  HYPOTHESES.md     All 14 hypotheses: prediction, result, conclusion
  EXPERIMENT_LOG.md Tabular log of every training run
  ARCHITECTURE.md   Key design decisions and tradeoffs
```

---

## Installation and Usage

**Requirements:** Python 3.12+

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train locally

```bash
python src/train.py
```

Trains for 1.5M steps with the TC curriculum. Logs to ClearML, saves checkpoints to `runs/checkpoints/`, best model to `runs/best_model/`. Automatically runs test-set evaluation on the best checkpoint at the end.

### Train on Modal (cloud)

```bash
.venv/bin/modal run src/modal_train.py
```

Runs on a 16-core Modal instance. Push your branch to GitHub first — Modal clones fresh from the branch set in `BRANCH`. Checkpoints are persisted to a Modal Volume; re-running after preemption auto-resumes.

### Evaluate

```bash
python src/evaluate.py
```

Loads `runs/best_model/best_model.zip` and runs a deterministic episode on the test split (2019–2025). Prints a metrics table with bootstrap CI and saves equity curve, rolling Sharpe, and weight heatmap plots to `runs/plots/`.

### Data splits

| Split | Period | Purpose |
|---|---|---|
| Train | 2000–2015 | Policy learning |
| Validation | 2015–2019 | Checkpoint selection during training |
| Test | 2019–2025 | Final evaluation — held out from all model selection |
