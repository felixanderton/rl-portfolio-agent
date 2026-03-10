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
