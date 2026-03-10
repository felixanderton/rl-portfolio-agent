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
**Status**: `[~]`
**Hypothesis**: `_A` and `_B` reset to zero at every `env.reset()`. The differential Sharpe denominator is `(B_prev - A_prev² + eps)^1.5`, which equals `eps^1.5` ≈ 0 for the first ~50-100 steps of each episode while the EMAs accumulate. This produces a degenerate reward signal at the start of every episode, poisoning a large fraction of the training data. Pre-warming the accumulators from the `window` steps of prior history already available in the feature matrix at `t=start` would make every step of every episode produce a valid reward.
**Change**: In `PortfolioEnv.reset()`, after sampling `_t`, compute `_A` and `_B` by running a forward pass over `features[_t-window:_t]` log-return columns using the EMA update rule before returning the first observation.
**Expected effect**: Cleaner reward signal throughout training, faster convergence, and higher final val Sharpe. The noisiness of the early-episode rewards has been diluting gradient quality across all of H1 and H2.
**Diagnostic**: Monitor reward/mean in ClearML — it should show a higher mean early in episodes. Val Sharpe curve should be less jagged.

---

## H5 — Vectorized training (n_envs=1 → n_envs=8)
**Status**: `[~]`
**Hypothesis**: Training on a single environment means every 2048-step rollout is one correlated trajectory from a single random start. The H1 val Sharpe curve oscillates ±0.04 for the last 500k steps — this is gradient variance, not a real plateau. Eight parallel environments with different random start points produce 8× more decorrelated transitions per PPO update, directly reducing gradient variance and smoothing the training signal.
**Change**: Replace the single `PortfolioEnv` + `Monitor` wrapper with a `make_vec_env` call using `n_envs=8`. Set `n_steps=2048` (per env), giving 16,384 transitions per update vs 2,048 now.
**Expected effect**: Smoother val Sharpe curve, faster convergence per wall-clock step, and a higher final val Sharpe as the policy escapes the noisy plateau seen in H1.
**Diagnostic**: The train/reward_std metric in ClearML should fall. The val Sharpe curve should oscillate less. If val Sharpe is unchanged, gradient variance was not the bottleneck.
**Note**: Best run after H3 — a larger network benefits more from the richer gradient signal.

---

## H6 — Multi-scale features (add 5-day and 60-day windows)
**Status**: `[ ]`
**Hypothesis**: The agent currently sees only 20-day log returns, volatility, and mean-reversion signals. Sector ETF momentum is strongest at 1-3 month horizons (60-day), and short-term reversals operate at weekly scales (5-day). These are genuine alpha signals the agent has no access to. Adding 5-day and 60-day versions of all three features per asset expands the feature set from 15 → 45 columns and gives the policy actionable signals at the timeframes where sector rotation actually works.
**Change**: In `data.py`, compute `_log_returns`, `_rolling_volatility`, and `_mean_reversion_signal` at `window=5` and `window=60` in addition to the existing `window=20`. Concatenate all three scales into the feature matrix. Update `PortfolioEnv.N_FEATURES_PER_ASSET` and observation size accordingly.
**Expected effect**: Substantial improvement in val Sharpe as the agent learns to exploit multi-horizon momentum and mean-reversion. This addresses a data constraint rather than a training constraint, so the ceiling is higher than H3/H5.
**Diagnostic**: Monitor per-asset allocation in ClearML — the agent should show stronger momentum-following behaviour (overweighting recent winners) once it can see 60-day signals.
**Note**: Largest change of the four. Run after H3 and H5 to establish a stable training baseline first.

---

## H11 — Expand asset universe (5 sector ETFs → 9 multi-asset ETFs)
**Status**: `[x]`
**Hypothesis**: All 5 current assets (XLK, XLE, XLF, XLV, XLI) are US equity sector ETFs — they are highly correlated and collapse together in risk-off environments. Adding TLT (long-term Treasuries), GLD (gold), EFA (international developed), and EEM (emerging markets) introduces genuinely uncorrelated return streams. The Sharpe ratio of the combined portfolio is fundamentally bounded by constituent correlations; with cross-asset diversification, the achievable Sharpe ceiling rises substantially. This is a change to the problem formulation, not a hyperparameter tweak. TRAIN_START shifts to 2005-01-01 (GLD constraint), reducing training data from ~3750 to ~2500 rows but gaining 4 uncorrelated asset classes.
**Change**: In `data.py`, update `TICKERS` to include TLT, GLD, EFA, EEM and set `TRAIN_START = "2005-01-01"`. In `environment.py`, derive `N_ASSETS` from `prices.shape[1]` rather than hardcoding 5. All observation/action space sizes update automatically.
**Expected effect**: Substantially higher val Sharpe as the agent learns to rotate into bonds and gold during equity drawdowns. This addresses the fundamental ceiling on the achievable Sharpe, not a training constraint.
**Diagnostic**: Monitor per-asset allocation in ClearML — the agent should show meaningful TLT/GLD allocation during stress periods. Val Sharpe should materially exceed H1's 0.5344.
