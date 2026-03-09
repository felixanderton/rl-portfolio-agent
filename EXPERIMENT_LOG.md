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

**ClearML task ID**: TBD

**Status**: Running
