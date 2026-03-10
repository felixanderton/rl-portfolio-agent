# Improvement Hypotheses

Baseline: val Sharpe **0.5240** — `lr=1e-4, n_steps=2048, ent_coef=0.01, 500k steps`

Status: `[ ]` untested · `[~]` in progress · `[x]` done

---

## H1 — Longer training to resolve policy collapse (500k → 1.5M steps)
**Status**: `[x]`
**Hypothesis**: Val Sharpe peaks at ~10% of training (0.38), collapses to 0.28 by 80%, then starts recovering — suggesting the policy hasn't converged rather than overfitting. The recovery trend at the end indicates more training budget would let it climb further. If it were overfitting, val Sharpe would plateau low rather than recover.
**Change**: Set `TOTAL_TIMESTEPS = 1_500_000`.
**Expected effect**: Val Sharpe continues the upward trend seen after 80% and exceeds 0.52 baseline.
**Diagnostic**: If val Sharpe keeps climbing → undertrained. If it collapses again → entropy collapse, move to H2.
**Result**: 0.5344 (baseline 0.5240, +1.9%). Peak of 0.5435 at step 1.2M, then oscillated and plateaued through 1.5M.
**Conclusion**: Confirmed. Longer training exceeded the baseline and validated the under-training diagnosis. However, val Sharpe plateaued rather than continuing to climb, suggesting the policy is near a local optimum with current entropy regularisation. H2 (ent_coef 0.01 → 0.05) is the natural next step to see if broader exploration can break through the plateau.

---

## H2 — Entropy coefficient to prevent policy collapse (0.01 → 0.05)
**Status**: `[x]`
**Hypothesis**: The val Sharpe dip from 0.38 to 0.28 during training coincides with entropy collapse — PPO squeezes the policy toward high-probability actions too early, causing over-commitment to a narrow allocation before a good local optimum is found. Higher entropy regularisation keeps the policy exploring longer.
**Prerequisite**: Run H1 first. Only test this if the dip persists with longer training.
**Change**: Set `ENT_COEF = 0.05`. Monitor `policy/weight_entropy` in ClearML — the dip should flatten.
**Expected effect**: Smoother val Sharpe curve, higher final val Sharpe.
**Result**: Stopped early at step 900k. Peak 0.5654 at step 50k (warm-start benefit), then steady degradation to 0.4035 by step 900k. Well below H1's 0.5344 and the 0.5240 baseline.
**Conclusion**: Disproven. Higher entropy regularisation destabilises the policy rather than helping it explore. The mid-training dip in H1 is likely noise or a natural saddle, not entropy collapse. ENT_COEF reverted to 0.01.

---

## H3 — Larger policy network ([64,64] → [256,256])
**Status**: `[x]`
**Hypothesis**: The default SB3 MlpPolicy uses a [64,64] network. With a 116-dimensional observation (20-day return history × 5 assets + vol + meanrev + weights + portfolio vol), the first layer compresses 116 inputs into 64 neurons before any cross-asset interactions can be learned. This is a binding capacity constraint — the policy cannot represent the patterns needed to beat 0.52. Increasing to [256,256] removes this bottleneck.
**Change**: Pass `policy_kwargs=dict(net_arch=[256, 256])` to the PPO constructor. All other hyperparameters unchanged (`lr=1e-4, n_steps=2048, ent_coef=0.01, total_timesteps=1_500_000`).
**Expected effect**: Higher final val Sharpe and smoother training curve as the network learns richer cross-asset representations.
**Diagnostic**: If val Sharpe improves significantly → capacity was the constraint. If val Sharpe is unchanged or worse → the problem is in the reward signal or data, not network size.
**Result**: Val Sharpe plateaued at ~0.36 after 500k steps — worse than [64,64] baseline. 2x slower per step.
**Conclusion**: Inconclusive in isolation. Larger network appears starved of diverse training data with n_envs=1. Shelved pending H5; will retest [256,256] once vectorized envs are in place.

---

## H4 — Fix EMA warm-up at episode start
**Status**: `[x]`
**Hypothesis**: `_A` and `_B` reset to zero at every `env.reset()`. The differential Sharpe denominator is `(B_prev - A_prev² + eps)^1.5`, which equals `eps^1.5` ≈ 0 for the first ~50-100 steps of each episode while the EMAs accumulate. This produces a degenerate reward signal at the start of every episode, poisoning a large fraction of the training data. Pre-warming the accumulators from the `window` steps of prior history already available in the feature matrix at `t=start` would make every step of every episode produce a valid reward.
**Change**: In `PortfolioEnv.reset()`, after sampling `_t`, compute `_A` and `_B` by running a forward pass over `features[_t-window:_t]` log-return columns using the EMA update rule before returning the first observation.
**Expected effect**: Cleaner reward signal throughout training, faster convergence, and higher final val Sharpe. The noisiness of the early-episode rewards has been diluting gradient quality across all of H1 and H2.
**Diagnostic**: Monitor reward/mean in ClearML — it should show a higher mean early in episodes. Val Sharpe curve should be less jagged.
**Result**: Final val Sharpe 0.6444. Peak 0.6564 at step 1,350,000. Clear acceleration after 950k steps (0.47 → 0.66 range). +20.6% vs H1 (0.5344). Now exceeds the momentum baseline on validation data.
**Conclusion**: Both fixes confirmed effective. The action space change unlocked concentrated positions; EMA warm-up eliminated degenerate early-episode rewards. H4 is the new best result and confirmed baseline for subsequent hypotheses. ClearML task ID: 06032dcd5f1947db86a11aa2450aa620.

---

## H11 — Expand asset universe (5 sector ETFs → 9 multi-asset ETFs)
**Status**: `[x]`
**Hypothesis**: All 5 current assets (XLK, XLE, XLF, XLV, XLI) are US equity sector ETFs — they are highly correlated and collapse together in risk-off environments. Adding TLT (long-term Treasuries), GLD (gold), EFA (international developed), and EEM (emerging markets) introduces genuinely uncorrelated return streams. The Sharpe ratio of the combined portfolio is fundamentally bounded by constituent correlations; with cross-asset diversification, the achievable Sharpe ceiling rises substantially. This is a change to the problem formulation, not a hyperparameter tweak. TRAIN_START shifts to 2005-01-01 (GLD constraint), reducing training data from ~3750 to ~2500 rows but gaining 4 uncorrelated asset classes.
**Change**: In `data.py`, update `TICKERS` to include TLT, GLD, EFA, EEM and set `TRAIN_START = "2005-01-01"`. In `environment.py`, derive `N_ASSETS` from `prices.shape[1]` rather than hardcoding 5. All observation/action space sizes update automatically.
**Expected effect**: Substantially higher val Sharpe as the agent learns to rotate into bonds and gold during equity drawdowns. This addresses the fundamental ceiling on the achievable Sharpe, not a training constraint.
**Diagnostic**: Monitor per-asset allocation in ClearML — the agent should show meaningful TLT/GLD allocation during stress periods. Val Sharpe should materially exceed H1's 0.5344.

---

## H5 — L2 weight decay on the PPO actor to reduce policy overfitting
**Status**: `[~]`
**Hypothesis**: The policy network has no weight regularisation — it is free to memorise training-period patterns with arbitrarily large weights. Adding L2 weight decay to the Adam optimiser used by SB3's MlpPolicy directly penalises large activations that encode training-specific patterns. The offline RL literature shows weight decay is the single most impactful individual regulariser on actor generalisation, reducing train/val performance gaps without destabilising training (unlike entropy coefficient increases, which H2 showed are harmful here).
**Change**: In `src/train.py`, pass `optimizer_kwargs=dict(weight_decay=1e-4)` inside `policy_kwargs` in the PPO constructor. This switches SB3 from Adam to AdamW, applying L2 decay to all policy parameters. No other changes.
**Expected effect**: Train Sharpe drops from ~4–5 toward ~1.5; val Sharpe holds at or near 0.6444 (H4 baseline), potentially improving ~5–8% based on the 6% average improvement reported across RL regularisation ablations.
**Diagnostic**: Monitor `policy/sharpe` (train) vs `validation/sharpe_ratio` gap — target gap <2.0 by end of training. Monitor `explained_variance` (must stay >0.5) and `clip_fraction` (should decrease from 0.3 as updates become more conservative).
**Falsification criterion**: If the train/val Sharpe gap remains above 3.0 at 1.5M steps, or `explained_variance` drops below 0.3, weight decay has not constrained the overfitting and may be harming the value function.
**Result**: Final val Sharpe 0.5211 (best checkpoint). Peak during training 0.4983 at step 1.0M. Curve oscillated between 0.39–0.50 throughout, never reaching H4 levels.
**Conclusion**: Disproven. Weight decay at 1e-4 substantially degraded performance (-18.4% vs H4 baseline of 0.6444). The regularisation appears to have over-constrained the policy, preventing it from learning meaningful concentrated allocations. The flat, noisy curve — contrasting with H4's late-training surge — suggests weight decay is interfering with the gradient dynamics that drove H4's improvement. Reverted to Adam (no weight decay). ClearML task ID: b465a5eb82524cf4971a1bcba02c095c.
**Note**: Source — Taiga et al., 2024. "The Role of Deep Learning Regularizations on Actors in Offline RL." arXiv:2409.07606.

---

## H6 — Transaction cost curriculum to suppress turnover-driven overfitting
**Status**: `[x]`
**Hypothesis**: The fixed `TRANSACTION_COST = 0.001` is too small to penalise the turnover (1.25) the policy achieves in training. Early in training, a small cost is intentional — the policy needs gradient signal before friction. But by 1M+ steps the cost should be high enough to actively deter the excessive rotation that allows train Sharpe to diverge to 4–5. Ramping transaction cost from ~0.0002 at step 0 to 0.001 at end of training (power-law schedule) gives the policy a free exploration phase and then progressively closes the gap between train and val conditions.
**Change**: In `src/train.py`, add a callback that updates `transaction_cost` on all vectorized envs at each checkpoint interval. In `src/environment.py`, expose `transaction_cost` as a settable attribute so the callback can update it mid-training.
**Expected effect**: `policy/turnover` plateaus below 0.5 by end of training. Train/val Sharpe gap narrows. Val Sharpe holds at or above 0.6444 while train Sharpe drops toward 1–2.
**Diagnostic**: Monitor `policy/turnover` (target: <0.5 by 1M steps), train vs val Sharpe gap, and `costs/mean_tx_cost_per_step` to confirm the ramp is taking effect.
**Falsification criterion**: If `policy/turnover` remains above 0.8 at 1.5M steps despite the ramp, or val Sharpe drops below 0.60, the curriculum has not constrained the overfit turnover.
**Result**: Final val Sharpe 0.7056 (post-training evaluation). Peak checkpoint 0.6928 at step 1,350,000. Clear late surge from 950k onwards: 0.5328 → 0.5967 → 0.6301 → 0.6928 → 0.6919 → 0.7056.
**Conclusion**: Confirmed. New best result — +9.5% vs H4 baseline (0.6444). The delayed cost ramp allows early exploration then progressively penalises turnover, reproducing H4's late-training surge at a higher level. H6 is the new confirmed baseline. ClearML task ID: 40f1afcadac442e2b78a0b40f6f72f01.
**Note**: Source — Soleymani & Mahootchi, 2025. "Regret-Optimized Portfolio Enhancement through Deep Reinforcement Learning and Future Looking Rewards." arXiv:2502.02619.

---

## H7 — Block bootstrap episode augmentation to prevent training-trajectory memorisation
**Status**: `[ ]`
**Hypothesis**: By 1M steps the policy has seen every trajectory in the 3750-row training window many times. Circular block bootstrap resampling generates synthetic episodes by stitching together contiguous blocks from the original data (block size ~80% of training length), preserving autocorrelation structure while preventing exact trajectory memorisation. Alternating every 10 episodes between real and bootstrapped data acts as explicit overfitting regularisation — the primary driver of out-of-sample Sharpe improvement in Soleymani & Mahootchi 2025.
**Change**: In `src/train.py`, add a `BlockBootstrapEnv` wrapper that, on every `reset()`, optionally replaces `self._features` and `self._prices` with a bootstrapped resample (circular block bootstrap, `block_size = 0.8 * T`). Alternate between real and bootstrapped episodes with probability 0.5.
**Expected effect**: Train Sharpe drops from ~4–5 toward 1–2. Val Sharpe holds or improves above H4's 0.6444, as the policy learns allocations that generalise across data perturbations.
**Diagnostic**: Monitor `policy/sharpe` (train) for sustained reduction. Val Sharpe should not degrade — if it drops below 0.55 early, block size is too small and is destroying the temporal structure the policy relies on.
**Falsification criterion**: If val Sharpe at 1.5M steps is below 0.60, bootstrap augmentation has harmed generalisation rather than helping — likely because synthetic episodes do not preserve the return autocorrelation the policy relies on.
**Note**: Source — Soleymani & Mahootchi, 2025. "Regret-Optimized Portfolio Enhancement through Deep Reinforcement Learning and Future Looking Rewards." arXiv:2502.02619.

---

## H9 — Episode length cap (252 steps / 1 trading year)
**Status**: `[~]`
**Hypothesis**: Current episodes run up to ~3700 steps — nearly the full training period. The policy can memorise a specific multi-year trajectory (dot-com crash → recovery → 2008 → rebound) as one continuous sequence, producing train Sharpe ~4–5. Capping episodes at 252 steps (1 trading year) means each episode sees a random 1-year window from the training data. The policy must learn allocations that work across many independent regimes rather than one long memorised path.
**Change**: Add `MAX_EPISODE_STEPS: int = 252` to `src/train.py` and pass it to `PortfolioEnv` via `_make_env`. In `src/environment.py`, add `max_episode_steps: int` parameter to `__init__`, store `self._episode_start` on `reset()`, and return `truncated=True` in `step()` when `self._t - self._episode_start >= self._max_episode_steps`.
**Expected effect**: Train Sharpe drops substantially from ~4–5. Val Sharpe holds or improves above H6's 0.7056.
**Diagnostic**: Monitor `policy/sharpe` (train) — expect significant drop toward 1–2. Monitor `validation/sharpe_ratio` — should still show a late-training surge. Monitor `reward/std` — shorter episodes should reduce within-episode variance.
**Falsification criterion**: If val Sharpe is below 0.65 at 1M steps, the cap is destroying the temporal signal the differential Sharpe EMA needs to stabilise.
