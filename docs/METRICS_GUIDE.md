# Metrics Guide

Reference for interpreting ClearML training diagnostics for this project.
Consult this before forming hypotheses — the goal is to diagnose the actual
bottleneck from the data before searching for solutions.

---

## Validation metrics

### `validation/sharpe_ratio` (logged every 50k steps)

The primary signal. Computed by running the current policy deterministically
on the val split (2015–2019) and evaluating with `evaluate_portfolio()` using
real prices. This is the ground truth for whether the policy is improving.

| Pattern | Diagnosis |
|---------|-----------|
| Rises steadily, then plateaus | Training budget exhausted — try more steps, or the policy is near a local optimum |
| Rises, collapses, recovers | Natural saddle point or entropy collapse mid-training — check `policy/weight_entropy` at the collapse point |
| Rises, then monotonically degrades | Overfitting to training period — check train vs val gap in `policy/sharpe` |
| Flat from the start (~0.35–0.45) | Policy not learning — check `reward/mean`, `costs/gross_return_per_step`, `explained_variance` |
| Late surge (only improves in the last third) | Was undertrained — run longer; check if it's still climbing at the end |
| Volatile throughout (oscillates ±0.05+) | High gradient variance — check `reward/std`, consider more envs or larger batch |

**Target**: beat the momentum baseline on val (currently ~0.52 on 2015–2019).
**Current best**: 0.7056 (H6 transaction cost curriculum, post-training eval).

---

## Policy metrics

### `policy/weight_entropy`

Shannon entropy of the portfolio weights at each step, averaged over the
episode: `H = -sum(w * log(w))`. For 5 assets:
- Maximum (equal weight): `log(5) ≈ 1.61`
- Fully concentrated (one asset): `0.0`

| Value | Meaning |
|-------|---------|
| Stays near 1.6 throughout | Policy never learns to differentiate — equal weight throughout. Check if reward signal is meaningful |
| Drops sharply early, stays low | Policy over-commits early, possibly due to high ent_coef being too low or a very strong early signal |
| Drops gradually over training | Healthy concentration learning — the policy is progressively learning which assets to favour |
| Drops then spikes back up | Policy collapsed and reset — check `approx_kl` at the same step |

### `policy/sharpe` (train-episode Sharpe)

Sharpe of the portfolio return within individual training episodes.
Compare against `validation/sharpe_ratio`:

- **Train ≈ Val**: no overfitting. If both are low, problem is in learning quality.
- **Train >> Val**: overfitting to the training period (2000–2015). Consider
  adding regularisation or shortening the training data window.
- **Train << Val**: unusual — could indicate that the random episode starts
  are hitting difficult market regimes disproportionately.

### `policy/turnover`

Mean L1 weight change per step. Roughly: 0.0 = frozen policy, 2.0 = full
rotation each step.

| Value | Meaning |
|-------|---------|
| < 0.05 | Policy barely moves — possibly converged to a fixed allocation |
| 0.05–0.3 | Healthy active management |
| > 0.5 | Excessive turnover — transaction costs are likely eating returns; check `costs/mean_tx_cost_per_step` |

---

## Reward metrics

### `reward/mean`

Mean portfolio return per step within episodes (z-score scale, not real
returns — see environment.py). Roughly tracks whether the agent is choosing
assets that go up.

- Near 0: the agent is not doing better than random allocation
- Positive trend over training: the agent is learning to pick winners

### `reward/std`

Standard deviation of per-step returns within an episode. High std = volatile
reward signal = noisy gradients.

- Should decrease or stabilise as training progresses
- If it stays high, check EMA denominator health (`_A`, `_B` accumulators) —
  a degenerate denominator amplifies reward noise

---

## Cost metrics

### `costs/gross_return_per_step` vs `costs/mean_tx_cost_per_step`

These should be compared together. If `mean_tx_cost` is close to or exceeds
`gross_return`, the policy is being consumed by friction.

| Situation | Implication |
|-----------|-------------|
| gross >> cost | Transaction costs are not the bottleneck |
| gross ≈ cost | Policy is marginal — turnover reduction may help (lower `transaction_cost` or add turnover penalty) |
| gross < cost | Policy is actively losing net of costs — likely a turnover/exploration problem |

---

## Asset allocation

### `asset_allocation/{ticker}` (XLK, XLE, XLF, XLV, XLI)

Mean weight assigned to each asset across the episode.

**What to look for:**
- Extreme concentration in one ticker (e.g. XLK > 0.7) suggests the policy
  has overfit to one asset's performance in the training period
- Uniform allocation (all ~0.2) means the policy hasn't learned differentiation
  — check `policy/weight_entropy`
- Economic plausibility: in a bull equity market (2000–2015 train period),
  XLK (tech) and XLV (healthcare) should receive above-average weight;
  XLE (energy) is cyclical and should vary more

---

## SB3 internal metrics (in the `train` ClearML graph)

These are logged automatically by stable-baselines3. They are diagnostic for
PPO training health, independent of portfolio performance.

| Metric | Healthy range | Problem if... |
|--------|---------------|---------------|
| `explained_variance` | Trending toward 0.8–1.0 | Stays near 0 → value function not learning; near -1 → value function actively wrong |
| `approx_kl` | 0.005–0.02 | Consistently > 0.05 → policy updating too aggressively (lower lr or increase clip_range); near 0 → not updating at all |
| `clip_fraction` | 0.05–0.2 | > 0.3 → too many clipped updates (reduce lr or n_steps); near 0 → policy change is negligible |
| `entropy_loss` | Negative, gradually approaching 0 | Spikes positive → policy collapsed to deterministic; drops very negative → policy is too random |
| `value_loss` | Decreasing over training | Plateaus high → value function can't fit the reward landscape; spikes → instability |
| `policy_gradient_loss` | Small negative | Large magnitude → strong policy gradient signal (can be good early, concerning late) |

---

## Diagnostic workflow

When reviewing a completed run, answer these questions in order:

1. **Did val Sharpe improve?** If not, skip to Q3.
2. **What was the val Sharpe curve shape?** (See patterns above.) What does it indicate?
3. **Is the reward signal healthy?** Check `reward/mean` trend and `reward/std` level.
4. **Is the value function learning?** Check `explained_variance` — if low, PPO's critic is the bottleneck.
5. **Is the policy concentrating?** Check `weight_entropy` trajectory.
6. **Is turnover reasonable?** Check `policy/turnover` vs `costs/mean_tx_cost_per_step`.
7. **Is one asset dominating?** Check `asset_allocation` — economic plausibility?
8. **Are PPO updates healthy?** Check `approx_kl` and `clip_fraction`.

Write down a one-sentence bottleneck hypothesis before looking at papers:
> *"The limiting factor appears to be X, evidenced by Y metric showing Z pattern."*
