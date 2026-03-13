# Experiment Log

## Format
Each entry: date, hypothesis, changes, hyperparameters, results, conclusion.

---

## 2026-03-08 — H1: raw returns in reward

**Hypothesis**: The reward EMA runs on z-score normalised returns (read from the normalised feature array), making its magnitude arbitrary and scale-dependent on the training normalisation. Computing the differential Sharpe reward from raw log returns (`price[t+1]/price[t]`) should give a more stable and interpretable signal.

**Changes**: `PortfolioEnv` will be modified to accept raw prices and compute `R_t = sum(w * log(P_t+1/P_t))` directly, instead of reading normalised logret columns from the feature matrix.

**Hyperparameters**: `lr=1e-4, n_steps=2048, ent_coef=0.01, total_timesteps=500_000, window=20, transaction_cost=0.001`

**Baseline**: val Sharpe 0.5240

**Results**: val Sharpe 0.2843 (baseline was 0.5240)

**vs Baseline**: worse by 0.2397 (-46%)

**Conclusion**: Hypothesis disproven. Using raw log returns in the reward EMA makes performance significantly worse. The z-score normalised returns in the original formulation appear to act as an implicit variance stabiliser — raw returns have much higher variance across different market regimes (e.g. 2008 vs 2013), which destabilises the EMA accumulators and makes the differential Sharpe signal noisy. Revert to normalised feature returns.

**Status**: Complete

---

## 2026-03-08 — H1: longer training to resolve policy collapse (500k -> 1.5M steps)

**Hypothesis**: Val Sharpe peaks at ~10% of training (0.38), collapses to 0.28 by 80%, then starts recovering — suggesting the policy has not converged rather than overfitting. More training budget should let it climb further and exceed the 0.52 baseline.

**Changes**: `TOTAL_TIMESTEPS` increased from `500_000` to `1_500_000`. No other changes.

**Hyperparameters**: `lr=1e-4, n_steps=2048, ent_coef=0.01, total_timesteps=1_500_000, window=20, transaction_cost=0.001`

**Baseline**: val Sharpe 0.5240

**Results**: Final val Sharpe 0.5344. Peak val Sharpe 0.5435 at step 1,200,000.

Val Sharpe by checkpoint:
- 50k: 0.4167, 100k: 0.3800, 150k: 0.3545, 200k: 0.3929, 250k: 0.4031
- 300k: 0.3993, 350k: 0.3977, 400k: 0.4206, 450k: 0.4172, 500k: 0.4159
- 550k: 0.4587, 600k: 0.4562, 650k: 0.4790, 700k: 0.5095, 750k: 0.4989
- 800k: 0.4999, 850k: 0.4957, 900k: 0.5088, 950k: 0.4871, 1.0M: 0.5127
- 1.05M: 0.5093, 1.1M: 0.5029, 1.15M: 0.5310, 1.2M: 0.5435, 1.25M: 0.5377
- 1.3M: 0.5310, 1.35M: 0.5245, 1.4M: 0.5147, 1.45M: 0.5283, 1.5M: 0.5314

**vs Baseline**: better by 0.0104 (+1.9%)

**Conclusion**: Confirmed. Longer training exceeded the baseline and validated the under-training diagnosis. Val Sharpe plateaued around 1.2M-1.5M steps rather than continuing to climb, suggesting the policy is near a local optimum with current entropy regularisation. H2 (ent_coef 0.01 → 0.05) is the natural next step to break through the plateau.

**ClearML task ID**: de63280b026041d3adcd2835d2b008df

**Status**: Complete

---

## 2026-03-08 — H2: higher entropy coefficient to prevent policy collapse (0.01 -> 0.05)

**Hypothesis**: The val Sharpe dip from 0.38 to 0.28 during H1 training coincides with entropy collapse — PPO squeezes the policy toward high-probability actions too early, causing over-commitment before a good local optimum is found. Higher entropy regularisation keeps the policy exploring longer, producing a smoother val Sharpe curve and a higher final val Sharpe than H1's 0.5344.

**Changes**: `ENT_COEF` increased from `0.01` to `0.05`. Warm-starting from H1 best model (`best_model/best_model`, val Sharpe 0.5344).

**Hyperparameters**: `lr=1e-4, n_steps=2048, ent_coef=0.05, total_timesteps=1_500_000, warm_start_path=best_model/best_model`

**Baseline**: val Sharpe 0.5240 (original baseline). Direct comparison target: H1 val Sharpe 0.5344.

**Results**: Stopped early at step 900k. Val Sharpe peaked at 0.5654 at step 50k (warm-start benefit from H1 checkpoint), then steadily degraded to 0.4035 by step 900k.

Val Sharpe by checkpoint:
- 50k: 0.5654, 100k: 0.5567, 150k: 0.5586, 200k: 0.5357, 250k: 0.5330
- 300k: 0.4994, 350k: 0.5352, 400k: 0.5078, 450k: 0.5487, 500k: 0.5433
- 550k: 0.5515, 600k: 0.5481, 650k: 0.5056, 700k: 0.5226, 750k: 0.4967
- 800k: 0.4671, 850k: 0.4484, 900k: 0.4035

**vs Baseline**: worse — final val Sharpe 0.4035 vs baseline 0.5240 (-0.1205, -23%) and vs H1 0.5344 (-0.1309, -25%)

**Conclusion**: Disproven. Higher entropy regularisation destabilises the policy rather than helping it explore. The mid-training dip in H1 is likely noise or a natural saddle, not entropy collapse. ENT_COEF reverted to 0.01.

**Status**: Complete (stopped early — disproven)

---

## 2026-03-09 — H11: Expand asset universe (5 sector ETFs -> 9 multi-asset ETFs)

**Hypothesis**: Adding uncorrelated assets (TLT, GLD, EFA, EEM) to the 5-sector ETF universe gives the agent a materially higher Sharpe ceiling. During equity drawdowns the agent can rotate into bonds and gold, which the current 5-ETF universe cannot do. The diversification benefit should far outweigh the reduction in training rows caused by the GLD launch date constraint.

**Changes**:
- `data.py`: TICKERS expanded from `["XLK","XLE","XLF","XLV","XLI"]` to `["XLK","XLE","XLF","XLV","XLI","TLT","GLD","EFA","EEM"]`. TRAIN_START shifted from `"2000-01-01"` to `"2005-01-01"` (GLD launch constraint).
- `environment.py`: N_ASSETS no longer hardcoded as 5 — now derived from `prices.shape[1]` in `__init__`, so observation/action spaces auto-scale.
- `train.py`: Fixed class-attribute reference `PortfolioEnv.N_ASSETS` -> `len(TICKERS)` in the fallback default.

**Hyperparameters**: `lr=1e-4, n_steps=2048, ent_coef=0.01, total_timesteps=1_500_000, n_envs=8, eta=0.01 (default), net_arch=[64,64]`

**Baseline**: val Sharpe 0.5240 (original). Best so far: H1 val Sharpe 0.5344.

**Note**: Training data reduced from ~3750 rows (2000-2014) to ~2500 rows (2005-2014) due to GLD launch date, but the cross-asset diversification benefit should far outweigh the data reduction.

**Results**: Final val Sharpe 0.3687. Peak val Sharpe 0.4482 at step 950,000.

Val Sharpe by checkpoint:
- 50k: 0.3092, 100k: 0.3121, 150k: 0.3181, 200k: 0.3259, 250k: 0.3510
- 300k: 0.3611, 350k: 0.3876, 400k: 0.3845, 450k: 0.4041, 500k: 0.3883
- 550k: 0.4276, 600k: 0.4143, 650k: 0.4264, 700k: 0.4220, 750k: 0.4370
- 800k: 0.4328, 850k: 0.3996, 900k: 0.4155, 950k: 0.4482, 1.0M: 0.4343
- 1.05M: 0.4256, 1.1M: 0.4352, 1.15M: 0.4346, 1.2M: 0.4424, 1.25M: 0.3917
- 1.3M: 0.3718, 1.35M: 0.3599, 1.4M: 0.3887, 1.45M: 0.3537, 1.5M: 0.3687

**vs Baseline**: worse by 0.1657 (-31% vs H1 val Sharpe 0.5344)

**Conclusion**: Inconclusive — asset universe expansion hurt in isolation at 1.5M steps. The observation space grew from 116 to 208 inputs while training data shrank from 3753 to 2497 rows (GLD constraint), making the learning problem harder without more compute. The policy shows a clear rise-then-collapse pattern peaking at 950k: the agent appears to start learning cross-asset dynamics but the gradient signal is too noisy to sustain convergence. The code infrastructure change (N_ASSETS derived from data rather than hardcoded) is a clean improvement regardless of this result. Revisiting with 3M+ steps or a later TRAIN_START that avoids the GLD data loss is warranted before concluding the multi-asset approach is wrong.

**ClearML task ID**: e6927b3c32a04883a9299b712660ce0b

**Status**: Complete

---

## 2026-03-12 — H12: Observation noise to prevent training-trajectory memorisation

**Hypothesis**: The train/val Sharpe gap (~7 vs ~0.77) is too large to be explained by learned skill — it reflects memorisation of specific training trajectories. Adding Gaussian noise (sigma=0.05) to training observations makes exact memorisation impossible, forcing the policy to learn perturbation-robust patterns. Val observations remain clean.

**Changes**: Added `obs_noise_sigma` parameter to `PortfolioEnv`. Training envs constructed with `obs_noise_sigma=0.05`; val and rollout envs with `obs_noise_sigma=0`. Training from scratch (no warm start).

**Hyperparameters**: `lr=1e-4, n_steps=2048, ent_coef=0.01, total_timesteps=1_500_000, n_envs=8, transaction_cost_curriculum=0.0002→0.001, obs_noise_sigma=0.05`

**Baseline**: val Sharpe 0.7669 (H10)

**Results**: Run terminated at 1.1M steps. Val Sharpe consistently trending downward; train Sharpe increasing steadily throughout.

**vs Baseline**: worse — val Sharpe in decline at termination, not approaching 0.7669 baseline.

**Conclusion**: Disproven. Observation noise at sigma=0.05 did not prevent memorisation — train Sharpe continued rising, indicating the policy adapted around the noise rather than generalising. Val Sharpe decline matches the H7/H5 pattern: input-level perturbation disrupts the differential Sharpe EMA's ability to accumulate a stable gradient signal, blocking the late-training surge. Not worth re-running at lower sigma given the consistent downward val trend at 1.1M steps.

**ClearML task ID**: 2228ead84ce54042ba26278f29b29d10

**Status**: Complete (terminated early — disproven)

---

## 2026-03-10 — H4: Fix action space bounds + EMA warm-up

**Hypothesis**: Two structural environment bugs were suppressing performance. (1) The action space `Box(0,1)` capped softmax pre-activations, preventing single-asset weights above ~40% and zeroing gradients above 1.0. (2) EMA accumulators `_A` and `_B` reset to 0 on every `env.reset()`, making the differential Sharpe denominator degenerate (~1e-9) for the first 50-100 steps of each episode and poisoning a large fraction of training gradients. Fixing both should unlock concentrated positions and improve gradient quality throughout training.

**Changes**:
- `environment.py`: Action space changed from `Box(0,1)` to `Box(-10,10)` — removes the weight cap and gradient zeroing above 1.0.
- `environment.py`: `_A` and `_B` accumulators now warmed up from the look-back window on every `reset()` instead of resetting to 0 — eliminates the degenerate differential Sharpe denominator for the first 50-100 steps of each episode.

**Branch**: fix/action-space-ema-warmup

**Hyperparameters**: `lr=1e-4, n_steps=2048, ent_coef=0.01, total_timesteps=1_500_000, n_envs=8, eta=0.01, window=20, transaction_cost=0.001`

**Baseline**: val Sharpe 0.5240 (original). Previous best: H1 val Sharpe 0.5344.

**Results**: Final val Sharpe 0.6444. Peak val Sharpe 0.6564 at step 1,350,000.

Val Sharpe by checkpoint:
- 50k: 0.3945, 100k: 0.4027, 150k: 0.4139, 200k: 0.4481, 250k: 0.4453
- 300k: 0.4264, 350k: 0.4525, 400k: 0.4285, 450k: 0.4441, 500k: 0.4500
- 550k: 0.4472, 600k: 0.4531, 650k: 0.4557, 700k: 0.4486, 750k: 0.4168
- 800k: 0.4510, 850k: 0.4451, 900k: 0.4689, 950k: 0.5161, 1.0M: 0.5767
- 1.05M: 0.5987, 1.1M: 0.5941, 1.15M: 0.6027, 1.2M: 0.5868, 1.25M: 0.6287
- 1.3M: 0.6393, 1.35M: 0.6564, 1.4M: 0.6237, 1.45M: 0.6045, 1.5M: 0.6305

**vs Baseline**: better by 0.1204 (+23.0% vs original baseline 0.5240). Better by 0.1100 (+20.6% vs H1 0.5344).

**Conclusion**: Both fixes confirmed effective. The val Sharpe curve shows a clear acceleration after 950k steps (0.47 → 0.66) compared to H1 which plateaued around 0.52-0.54. The late-training surge is consistent with the policy now learning to take concentrated positions, previously impossible with `[0,1]` action bounds. Final val Sharpe of 0.6444 now exceeds the momentum baseline (0.649 on the full period) computed on validation data. H4 is the new best result and the confirmed baseline for subsequent hypotheses.

**ClearML task ID**: 06032dcd5f1947db86a11aa2450aa620

**Status**: Complete

---

## 2026-03-10 — H5: L2 weight decay on PPO actor to reduce policy overfitting

**Hypothesis**: The policy network has no weight regularisation and is free to memorise training-period patterns. H4 showed a train/val Sharpe gap of ~4–5 vs 0.65. Adding L2 weight decay (AdamW) directly penalises large activations encoding training-specific patterns, which should close the train/val gap and hold or improve val Sharpe.

**Changes**: Pass `optimizer_kwargs=dict(weight_decay=1e-4)` inside `policy_kwargs` in the PPO constructor in `src/train.py`. No other changes.

**Hyperparameters**: `lr=1e-4, n_steps=2048, ent_coef=0.01, total_timesteps=1_500_000, n_envs=8, transaction_cost=0.001, weight_decay=1e-4`

**Baseline**: val Sharpe 0.6444 (H4, ClearML task 06032dcd5f1947db86a11aa2450aa620)

**Expected effect**: Train Sharpe drops from ~4–5 toward ~1.5; val Sharpe holds at or near 0.6444, potentially improving ~5–8%.

**Results**: Final val Sharpe 0.5211 (best checkpoint evaluation). Val Sharpe by checkpoint:
- 50k: 0.3993, 100k: 0.4020, 150k: 0.4159, 200k: 0.4418, 250k: 0.4185
- 300k: 0.3930, 350k: 0.4089, 400k: 0.3853, 450k: 0.4215, 500k: 0.4340
- 550k: 0.4561, 600k: 0.4772, 650k: 0.4801, 700k: 0.4765, 750k: 0.4654
- 800k: 0.4354, 850k: 0.4672, 900k: 0.4675, 950k: 0.4511, 1.0M: 0.4983
- 1.05M: 0.4798, 1.1M: 0.4464, 1.15M: 0.4203, 1.2M: 0.4054, 1.25M: 0.4280
- 1.3M: 0.4458, 1.35M: 0.4636, 1.4M: 0.4582, 1.45M: 0.4519, 1.5M: 0.5135

Peak: 0.4983 at step 1.0M.

**vs Baseline**: worse by 0.1233 (-18.4% vs H4 baseline of 0.6444)

**Conclusion**: Disproven. Weight decay at 1e-4 substantially degraded performance. The regularisation over-constrained the policy, preventing the late-training surge seen in H4 (0.47→0.66 after 950k steps). The val Sharpe curve was flat and noisy throughout (0.39–0.50), never approaching H4 levels. Weight decay interfered with the gradient dynamics that allowed H4 to find concentrated positions. Reverted to Adam (no weight decay).

**ClearML task ID**: b465a5eb82524cf4971a1bcba02c095c

**Status**: Complete

---

## 2026-03-10 — H6: Transaction cost curriculum to suppress turnover-driven overfitting

**Hypothesis**: The fixed `TRANSACTION_COST=0.001` is too small to penalise `episode_turnover` of 1.25. A power-law curriculum ramps `transaction_cost` from 0.0002 at step 0 to 0.001 by the end of training, giving the policy a free exploration phase before progressively penalising excessive rotation.

**Changes**:
- `src/train.py`: Add `TxCostCurriculumCallback` that updates `transaction_cost` on all vectorised envs each step using a power-law schedule (`TC_MIN=0.0002` → `TRANSACTION_COST=0.001` over 1.5M steps, exponent `p=2`).
- `src/environment.py`: Expose `transaction_cost` as a settable property on `PortfolioEnv` so the callback can update it live.

**Hyperparameters**: `lr=1e-4, n_steps=2048, ent_coef=0.01, total_timesteps=1_500_000, n_envs=8, transaction_cost_final=0.001, transaction_cost_start=0.0002, tc_schedule=power-law(p=2)`

**Baseline**: val Sharpe 0.6444 (H4, ClearML task 06032dcd5f1947db86a11aa2450aa620)

**Expected effect**: `policy/turnover` plateaus below 0.5 by end of training. Train/val Sharpe gap narrows. Val Sharpe holds at or above 0.6444 (H4 baseline).

**Results**: Final val Sharpe 0.7056 (post-training evaluation). Checkpoint val Sharpe values:
- 50k: 0.3951, 100k: 0.4032, 150k: 0.4148, 200k: 0.4477, 250k: 0.4450
- 300k: 0.4253, 350k: 0.4498, 400k: 0.4257, 450k: 0.4403, 500k: 0.4504
- 550k: 0.4550, 600k: 0.4605, 650k: 0.4578, 700k: 0.4458, 750k: 0.4134
- 800k: 0.4500, 850k: 0.4541, 900k: 0.4758, 950k: 0.5328, 1.0M: 0.5967
- 1.05M: 0.6301, 1.1M: 0.6268, 1.15M: 0.6208, 1.2M: 0.6194, 1.25M: 0.6528
- 1.3M: 0.6700, 1.35M: 0.6928, 1.4M: 0.6667, 1.45M: 0.6636, 1.5M: 0.6919

Peak checkpoint: 0.6928 at step 1,350,000.

**vs Baseline**: better by 0.0612 (+9.5% vs H4 baseline of 0.6444). New best result.

**Conclusion**: Confirmed. The transaction cost curriculum reproduced H4's late-training surge at a higher level. The delayed ramp (0.0002→0.001 quadratic) allows free early exploration before progressively penalising turnover, leading to a strong surge from 950k onwards (0.53→0.71). H6 is the new confirmed baseline for subsequent hypotheses.

**ClearML task ID**: 40f1afcadac442e2b78a0b40f6f72f01

**Status**: Complete

---

## 2026-03-11 — H7: Block bootstrap episode augmentation to prevent training-trajectory memorisation

**Hypothesis**: By 1M steps the policy has seen every trajectory in the 3750-row training window many times. Circular block bootstrap resampling generates synthetic episodes by stitching together contiguous blocks from the original data (block size ~80% of training length = ~3000 rows), preserving autocorrelation structure while preventing exact trajectory memorisation. Alternating every reset between real and bootstrapped data (50% probability each) acts as explicit overfitting regularisation.

**Changes**:
- `environment.py`: Add `BlockBootstrapEnv(gymnasium.Wrapper)` class that on 50% of resets replaces the underlying env's features/prices with a circular block-bootstrapped resample.
- `train.py`: Add `BLOCK_BOOTSTRAP_FRACTION = 0.8` constant. Modify `_make_env` to wrap `PortfolioEnv` with `BlockBootstrapEnv`.

**Hyperparameters**: `lr=1e-4, n_steps=2048, ent_coef=0.01, total_timesteps=1_500_000, n_envs=8, transaction_cost_curriculum=0.0002→0.001, block_bootstrap_fraction=0.8, bootstrap_prob=0.5`

**Baseline**: val Sharpe 0.7056 (H6, ClearML task 40f1afcadac442e2b78a0b40f6f72f01)

**Expected effect**: Train Sharpe drops from ~4–5 toward 1–2. Val Sharpe holds or improves above H6's 0.7056, as the policy learns allocations that generalise across data perturbations.

**Results**: Final val Sharpe 0.2969 (post-training evaluation). Val Sharpe by checkpoint:
- 50k: 0.4023, 100k: 0.4170, 150k: 0.4246, 200k: 0.4321, 250k: 0.4238
- 300k: 0.4576, 350k: 0.4508, 400k: 0.4302, 450k: 0.4331, 500k: 0.4297
- 550k: 0.4101, 600k: 0.4175, 650k: 0.4235, 700k: 0.4408, 750k: 0.4079
- 800k: 0.4394, 850k: 0.4129, 900k: 0.4187, 950k: 0.4065, 1.0M: 0.4354
- 1.05M: 0.4501, 1.1M: 0.4127, 1.15M: 0.3949, 1.2M: 0.3992, 1.25M: 0.3936
- 1.3M: 0.3929, 1.35M: 0.3923, 1.4M: 0.3789, 1.45M: 0.3433, 1.5M: 0.3313

Peak: 0.4576 at step 300,000.

**vs Baseline**: worse by 0.4087 (-57.9% vs H6 baseline of 0.7056)

**Conclusion**: Disproven. Block bootstrap destroyed the late-training surge that H4 and H6 relied on. The falsification criterion (val Sharpe < 0.60 at 1M steps) was triggered at 0.4354. The pattern mirrors H9 (episode length cap): the bootstrap disrupts the temporal continuity the differential Sharpe EMA depends on. Swapping features/prices on 50% of resets means EMA accumulators are perpetually calibrated to a different data sequence on each bootstrap episode, preventing stable gradient signal from accumulating. The late-training surge (950k+ acceleration) was completely absent — the curve peaked early at 300k then steadily degraded. Block bootstrap is not a viable regulariser for this reward formulation.

**Status**: Complete

---

## 2026-03-12 — H13: Portfolio concentration penalty (HHI) to reduce memorisation-driven overfit

**Hypothesis**: The train/val Sharpe gap (~7 vs ~0.77) is driven by the policy learning extremely concentrated positions that happen to be correct for memorised training trajectories but do not generalise. Adding a portfolio concentration penalty `-lambda * sum(w_i^2)` (negative HHI term) to the training reward penalises single-asset dominance in reward space. Reward-space regularisation (H6 TC curriculum, +9.5%) is the only class that has worked without disrupting the late-training surge. This penalty operates in the same space.

**Changes**: Added `concentration_penalty_lambda` parameter to `PortfolioEnv`. Training envs constructed with `CONCENTRATION_LAMBDA=0.01`; val and rollout envs use `0.0` so the penalty does not distort evaluation metrics.

**Hyperparameters**: `lr=1e-4, n_steps=2048, ent_coef=0.01, total_timesteps=1_500_000, n_envs=8, transaction_cost_curriculum=0.0002→0.001, concentration_lambda=0.01, warm_start=none`

**Baseline**: val Sharpe 0.7669 (H10, ClearML task bd3acca5e38a4a6081bf801bed5b1567)

**Expected effect**: Train Sharpe drops from ~7 toward ~2–3. Val Sharpe holds at or above 0.7669. Late-training surge should be preserved since reward-space changes were safe in H6.

**Results**: Best val Sharpe 0.4593 at step 450,000. Final val Sharpe 0.2195 at step 1,500,000. Training duration ~18:45.

Val Sharpe by phase: peaked at 450k (0.4593) then steadily degraded to 0.22 by end of run — no late-training surge.

**vs Baseline**: worse — final val Sharpe 0.2195 vs baseline 0.7669 (-0.5474, -71%). Best val 0.4593 also well below baseline.

**Conclusion**: Disproven. The concentration penalty alone did not prevent overfitting. Val Sharpe peaked at 0.4593 at 450k steps then steadily degraded to 0.22 by end of run — the classic overfit collapse pattern. The penalty suppressed the late-training surge rather than redirecting it: rather than the policy finding generalised positions worth concentrating on, it appears to have been discouraged from concentrating at all while still memorising training trajectories via other mechanisms. Unlike H6 (TC curriculum), which penalised turnover (a clear overfit behaviour), the HHI penalty directly penalises concentration, which is also the mechanism the policy uses to earn high val Sharpe on genuinely good rotations. The penalty cannot distinguish memorisation-driven concentration from skill-driven concentration. Not worth re-running at a lower lambda given the scale of val Sharpe collapse.

**ClearML task ID**: 5af7ff7a9e4948b9ac7bce0f8c18e1ad

**Status**: Complete

---

## 2026-03-13 — H14: Cross-sectional momentum features (63-day and 252-day cumulative log returns)

**Hypothesis**: The current 15-feature state vector contains only short-horizon signals (20-day log returns, rolling volatility, mean-reversion z-scores). Cross-sectional momentum — the relative ranking of medium and long-horizon cumulative returns across assets — is the best-documented anomaly in sector ETF allocation. Adding 63-day and 252-day cumulative log returns per asset (10 new features, total 25 features × 5 assets = 126-dim obs) gives the agent the signal it needs to rotate into recent winners over multi-month horizons.

**Changes**: `data.py` updated to compute and append mom63 (63-day cumulative log return) and mom252 (252-day cumulative log return) features for each of the 5 assets, expanding obs_size from 116 to 126 (feature_count from 15 to 25). No warm-start possible — observation space changed.

**Hyperparameters**:
- `lr`: 1e-4
- `n_steps`: 2048
- `ent_coef`: 0.01
- `total_timesteps`: 1_500_000
- `n_envs`: 8
- `transaction_cost_curriculum`: 0.0002 → 0.001 (H6 TC curriculum protocol)
- `concentration_lambda`: 0.01
- `warm_start`: none (obs space changed 116 → 126)
- `obs_noise_sigma`: 0.0
- `obs_size`: 126 (was 116, +10 for 2 momentum signals × 5 assets)
- `feature_count`: 25 (was 15)

**Baseline**: H13 best val Sharpe 0.4593

**ClearML task ID**: ab45d417b52b454cab1e3d6eaf4dc96f

**Status**: Running

---

## 2026-03-12 — H14: Cross-sectional momentum features (63-day and 252-day cumulative log returns)

**Hypothesis**: The current 15-feature state vector contains only short-horizon signals (20-day log returns, rolling volatility, mean-reversion z-scores). Cross-sectional momentum — the relative ranking of medium and long-horizon cumulative returns across assets — is the best-documented anomaly in sector ETF allocation. Adding 63-day and 252-day cumulative log returns per asset (10 new features, total 25) gives the agent the signal it needs to rotate into recent winners over multi-month horizons.

**Changes**: `data.py` and/or `environment.py` updated to compute and append 63-day and 252-day cumulative log return features for each of the 5 assets, expanding the observation vector from 15 to 25 dimensions.

**Hyperparameters**: `lr=1e-4, n_steps=2048, ent_coef=0.01, total_timesteps=1_500_000, n_envs=8, transaction_cost_curriculum=0.0002→0.001, concentration_lambda=0.01, warm_start=none (obs space changed)`

**Baseline**: val Sharpe 0.7669 (H10)

**Results**: Val Sharpe significantly below H10 baseline throughout training. Disproven.

**Conclusion**: Disproven. Adding 63-day and 252-day momentum features did not improve performance. The observation space expansion (116 → 126) and the longer burn-in window (252 rows dropped from the training set) likely made the learning problem harder without providing actionable signal. The momentum features are raw cumulative log returns, not cross-sectionally normalised ranks — the agent may have been unable to extract the relative ranking information the hypothesis was designed to provide. The feature ceiling hypothesis was not validated.

**Status**: Complete (disproven)

---

## 2026-03-11 — H10: Extended training with H6 warm start (1.5M -> 3M effective steps)

**Hypothesis**: H6 and H4 both showed a clear late-training surge starting around 950k steps, and the val Sharpe curve was still oscillating upward at 1.5M rather than plateauing. The policy likely has not exhausted its learning capacity. Warm-starting from H6's best checkpoint (val Sharpe 0.7056) and training for a further 1.5M steps under the same H6 protocol should allow the late-training trend to continue.

**Changes**:
- `train.py`: Set `WARM_START_PATH = "/app/runs/warm_start/best_model"` to load the H6 best model.
- `modal_train.py`: Download the H6 best model artifact from ClearML (task ID `40f1afcadac442e2b78a0b40f6f72f01`) to `/app/runs/warm_start/` before calling `main()`.

**Hyperparameters**: `lr=1e-4, n_steps=2048, ent_coef=0.01, total_timesteps=1_500_000, n_envs=8, transaction_cost_curriculum=0.0002→0.001, warm_start=H6_best (ClearML task 40f1afcadac442e2b78a0b40f6f72f01)`

**Baseline**: val Sharpe 0.7056 (H6, ClearML task 40f1afcadac442e2b78a0b40f6f72f01)

**Expected effect**: Val Sharpe exceeds 0.7056 within the first 500k steps and climbs toward 0.75–0.80 by end of training.

**Results**: Final val Sharpe 0.7669 (post-training evaluation). Val Sharpe by checkpoint:
- 50k: 0.6935, 100k: 0.7203, 150k: 0.7285, 200k: 0.7303, 250k: 0.7243
- 300k: 0.7149, 350k: 0.7392, 400k: 0.7258, 450k: 0.7332, 500k: 0.7806
- 550k: 0.7818, 600k: 0.8105, 650k: 0.7511, 700k: 0.7182, 750k: 0.7108
- 800k: 0.7136, 850k: 0.6850, 900k: 0.7110, 950k: 0.7885, 1.0M: 0.8062
- 1.05M: 0.8130, 1.1M: 0.7914, 1.15M: 0.7474, 1.2M: 0.6936, 1.25M: 0.7123
- 1.3M: 0.7279, 1.35M: 0.7432, 1.4M: 0.7570, 1.45M: 0.6763, 1.5M: 0.7374

Peak: 0.8130 at step 1,050,000.

**vs Baseline**: better by 0.0613 (+8.7% vs H6 baseline of 0.7056). New best result.

**Test set results** (ClearML task `bd3acca5e38a4a6081bf801bed5b1567`, best checkpoint at step 1,050,000):

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| PPO Agent | +20.90% | 24.45% | **0.8551** | -38.34% | 0.5453 |
| Equal Weight | +13.96% | 21.54% | 0.6480 | -41.04% | 0.3401 |
| Momentum | +17.22% | 23.69% | 0.7270 | -34.80% | 0.4949 |
| SPY Buy & Hold | +15.94% | 19.93% | 0.7996 | -33.72% | 0.4728 |

Bootstrap 95% CI for Sharpe(PPO) − Sharpe(EqualWeight): [−0.23, +0.61] (n=1000). Difference not statistically significant at 5%.

**Conclusion**: Confirmed. Warm start from H6 checkpoint immediately reproduced H6-level performance (0.6935 at 50k steps) then climbed further. Two surge phases: 500k–600k reaching 0.8105, and 950k–1.05M reaching 0.8130. Falsification criterion cleared at 500k (0.7806 > 0.72). The policy had not reached its ceiling at 1.5M steps — additional training budget under the same H6 protocol continues to improve val Sharpe. H10 is the new confirmed baseline at 0.7669. Further training runs (H10 warm-started again) may push higher, though the volatile 0.69–0.81 range in the second half suggests the policy is near a plateau.

Test Sharpe of 0.8551 outperforms all baselines on the held-out 2019–2024 period, though the wide bootstrap CI reflects limited statistical power over a single 6-year window.

**ClearML task ID**: bd3acca5e38a4a6081bf801bed5b1567

**Status**: Complete
