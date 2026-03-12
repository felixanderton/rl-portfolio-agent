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
**Status**: `[x]`
**Hypothesis**: By 1M steps the policy has seen every trajectory in the 3750-row training window many times. Circular block bootstrap resampling generates synthetic episodes by stitching together contiguous blocks from the original data (block size ~80% of training length), preserving autocorrelation structure while preventing exact trajectory memorisation. Alternating every 10 episodes between real and bootstrapped data acts as explicit overfitting regularisation — the primary driver of out-of-sample Sharpe improvement in Soleymani & Mahootchi 2025.
**Change**: In `src/train.py`, add a `BlockBootstrapEnv` wrapper that, on every `reset()`, optionally replaces `self._features` and `self._prices` with a bootstrapped resample (circular block bootstrap, `block_size = 0.8 * T`). Alternate between real and bootstrapped episodes with probability 0.5.
**Expected effect**: Train Sharpe drops from ~4–5 toward 1–2. Val Sharpe holds or improves above H4's 0.6444, as the policy learns allocations that generalise across data perturbations.
**Diagnostic**: Monitor `policy/sharpe` (train) for sustained reduction. Val Sharpe should not degrade — if it drops below 0.55 early, block size is too small and is destroying the temporal structure the policy relies on.
**Falsification criterion**: If val Sharpe at 1.5M steps is below 0.60, bootstrap augmentation has harmed generalisation rather than helping — likely because synthetic episodes do not preserve the return autocorrelation the policy relies on.
**Result**: Final val Sharpe 0.2969. Peak 0.4576 at step 300,000. Curve degraded steadily after 300k — no late-training surge.
**Conclusion**: Disproven. Block bootstrap destroyed the late-training surge. Falsification criterion triggered at 1M steps (0.4354 < 0.60). Same root cause as H9: disrupts the temporal continuity the differential Sharpe EMA depends on. Not a viable regulariser for this reward formulation.
**Note**: Source — Soleymani & Mahootchi, 2025. "Regret-Optimized Portfolio Enhancement through Deep Reinforcement Learning and Future Looking Rewards." arXiv:2502.02619.

---

## H12 — Observation noise to prevent training-trajectory memorisation
**Status**: `[x]`
**Hypothesis**: Diagnostic plots show the agent earning ~40% cumulative return over training windows where equal-weight earns ~5%, while val performance is near equal-weight. This gap is too large to be explained by learned skill — it is memorisation of specific training trajectories. With 8 envs × 1.5M steps across ~3750 training rows, each row is seen ~3200 times; the policy has effectively memorised the exact feature values associated with profitable rotations. Increasing transaction costs cannot close a 40% gap. The most direct fix is to make exact memorisation impossible: adding small Gaussian noise to the observation at each training step forces the agent to learn policies that are robust to perturbations of feature values rather than keyed to specific numbers. Val observations remain clean (noise=0), so any performance improvement reflects genuine generalisation rather than adaptation to noise.
**Change**: Add a `obs_noise_sigma` parameter to `PortfolioEnv`. When non-zero, add `np.random.normal(0, sigma, obs.shape)` to the observation returned by `_get_obs()` during training. Set `sigma=0.05` (5% of a typical z-score unit). Val and rollout envs are constructed with `obs_noise_sigma=0`. Warm-start from H10 best model.
**Hyperparameters**: `lr=1e-4, n_steps=2048, ent_coef=0.01, total_timesteps=1_500_000, n_envs=8, transaction_cost_curriculum=0.0002→0.001, warm_start=none, obs_noise_sigma=0.05`
**Expected effect**: Train Sharpe drops from ~7 toward ~1–2 as memorised patterns are disrupted. Val Sharpe holds at or above 0.7669 (H10 baseline) — if the previous val performance was partly driven by generalised patterns rather than pure memorisation, noise regularisation should preserve or improve it.
**Diagnostic**: Compare `train_event_zoom` vs `event_zoom` plots in ClearML — training cumulative value should no longer massively outpace equal-weight. Monitor `policy/sharpe` (train) for reduction and `validation/sharpe_ratio` for stability.
**Falsification criterion**: If val Sharpe drops below 0.70 by 750k steps, the noise level is destroying useful signal rather than just preventing memorisation. Try sigma=0.01 before abandoning.
**Result**: Run terminated at 1.1M steps. Val Sharpe was consistently trending downward throughout; train Sharpe was increasing steadily — the opposite of the desired effect.
**Conclusion**: Disproven. Observation noise at sigma=0.05 did not prevent memorisation — it degraded the gradient signal to the point where the policy could not learn useful val-generalising patterns. Train Sharpe continued rising, indicating the policy adapted around the noise rather than being forced to generalise. The val Sharpe decline mirrors H7 (block bootstrap) and H5 (weight decay): input-level perturbation disrupts the differential Sharpe EMA's ability to accumulate a stable signal, preventing the late-training surge. Not worth re-running at a lower sigma given the consistent downward val trend at 1.1M steps. ClearML task ID: 2228ead84ce54042ba26278f29b29d10.

---

## H13 — Portfolio concentration penalty to reduce memorisation-driven overfit
**Status**: `[ ]`
**Hypothesis**: The train/val Sharpe gap (~7 vs ~0.77) is driven by the policy learning to take extremely concentrated positions that happen to be correct for memorised training trajectories but don't generalise. H5 (weight decay) and H12 (observation noise) both failed because they interfered with the gradient dynamics of the late-training surge. H6 showed that reward-space regularisation is safe — it produced +9.5% without disrupting the surge. Adding a portfolio concentration penalty (negative HHI term: `-lambda * sum(w_i^2)`) directly penalises the mechanism of overfit rather than the gradients or inputs, and operates in the same reward space that H6 successfully used.
**Change**: In `PortfolioEnv.step()`, subtract `concentration_penalty_lambda * np.sum(weights**2)` from the reward. Add `CONCENTRATION_LAMBDA` constant to `train.py`. No other changes.
**Hyperparameters**: `lr=1e-4, n_steps=2048, ent_coef=0.01, total_timesteps=1_500_000, n_envs=8, transaction_cost_curriculum=0.0002→0.001, concentration_lambda=0.01, warm_start=none`
**Baseline**: val Sharpe 0.7669 (H10, ClearML task bd3acca5e38a4a6081bf801bed5b1567)
**Expected effect**: Train Sharpe drops from ~7 toward ~2–3 as the policy is penalised for the concentrated positions it uses to exploit memorised trajectories. Val Sharpe holds at or above 0.7669 — generalised allocation patterns are less concentrated and should be less penalised.
**Diagnostic**: Monitor `policy/hhi` (log `np.sum(weights**2)` per step) in ClearML — target mean HHI < 0.4 by end of training vs current ~0.8+. Monitor train vs val Sharpe gap for narrowing. The late-training surge pattern (acceleration after ~950k) should be preserved since reward-space changes did not disrupt it in H6.
**Falsification criterion**: If val Sharpe drops below 0.70 by 750k steps, the penalty is suppressing useful concentrated positions rather than just memorisation-driven ones — try lambda=0.005.

---

## H10 — Extended training with H6 warm start (1.5M → 3M effective steps)
**Status**: `[x]`
**Hypothesis**: H6 and H4 both showed a clear late-training surge starting around 950k steps, and the val Sharpe curve was still oscillating upward at 1.5M rather than plateauing. The policy likely has not exhausted its learning capacity — it just ran out of training budget. Warm-starting from H6's best checkpoint (val Sharpe 0.7056) and training for a further 1.5M steps under the same H6 protocol (TC curriculum, same hyperparameters) should allow the late-training trend to continue and push val Sharpe materially above 0.70.
**Change**: Set `WARM_START_PATH` in `train.py` to load the H6 best model. In `modal_train.py`, download the H6 artifact from ClearML (task ID `40f1afcadac442e2b78a0b40f6f72f01`) before calling `main()`. All other hyperparameters unchanged from H6.
**Hyperparameters**: `lr=1e-4, n_steps=2048, ent_coef=0.01, total_timesteps=1_500_000, n_envs=8, transaction_cost_curriculum=0.0002→0.001, warm_start=H6_best`
**Expected effect**: Val Sharpe exceeds 0.7056 within the first 500k steps (warm-start benefit) and climbs toward 0.75–0.80 by end of training.
**Diagnostic**: Val Sharpe at 50k steps should be near or above 0.70 (warm-start benefit from H6). A drop below 0.60 at 50k steps would indicate the TC curriculum reset is destabilising the loaded policy.
**Falsification criterion**: If val Sharpe does not exceed 0.72 by 750k steps, the policy has reached a true local optimum with the current architecture and reward formulation.
**Result**: Final val Sharpe 0.7669 (post-training evaluation). Peak 0.8130 at step 1,050,000. Val Sharpe by checkpoint:
- 50k: 0.6935, 100k: 0.7203, 150k: 0.7285, 200k: 0.7303, 250k: 0.7243
- 300k: 0.7149, 350k: 0.7392, 400k: 0.7258, 450k: 0.7332, 500k: 0.7806
- 550k: 0.7818, 600k: 0.8105, 650k: 0.7511, 700k: 0.7182, 750k: 0.7108
- 800k: 0.7136, 850k: 0.6850, 900k: 0.7110, 950k: 0.7885, 1.0M: 0.8062
- 1.05M: 0.8130, 1.1M: 0.7914, 1.15M: 0.7474, 1.2M: 0.6936, 1.25M: 0.7123
- 1.3M: 0.7279, 1.35M: 0.7432, 1.4M: 0.7570, 1.45M: 0.6763, 1.5M: 0.7374
**Conclusion**: Confirmed. New best result — +8.7% vs H6 (0.7056). Warm start benefit immediately visible: 0.6935 at 50k vs H6's 0.38–0.45 range at the same point. Falsification criterion cleared at 500k (0.7806 > 0.72). Two surge phases: 500k–600k (0.78→0.81) and 950k–1.05M (0.79→0.81). Val Sharpe remains volatile in the 0.69–0.81 range throughout — consistent with the policy exploring concentrated positions. H10 is the new confirmed baseline.
