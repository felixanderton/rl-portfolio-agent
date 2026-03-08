Build me a deep reinforcement learning portfolio allocation project in Python. Here is the full specification:
Assets and Data

Use yfinance to download adjusted close prices for 5 sector ETFs: XLK, XLE, XLF, XLV, XLI
Chronological train/val/test split: train 2000–2015, validation 2015–2019, test 2020–2024
Compute log returns, 20-day rolling volatility, and rolling mean-reversion signal (position relative to rolling mean)
Normalise all features using statistics computed only on the training set — no data leakage

File structure

data.py — data download, feature engineering, train/val/test splits, outputs clean numpy arrays
environment.py — custom Gymnasium environment
baselines.py — equal weight, momentum (12-1 month), buy-and-hold SPY, shared evaluation function
train.py — PPO training script using stable-baselines3
evaluate.py — evaluation, metrics table, plots
ablation.py — ablation study across three variants

Environment (environment.py)

Subclass gymnasium.Env
State vector: per-asset log returns (20-day window), rolling volatility, mean-reversion signal, plus current portfolio weights and current portfolio volatility — flattened into a single vector
Action space: Box(0, 1, shape=(5,)) — continuous weights normalised to sum to 1 via softmax inside the step function
Reward: Differential Sharpe Ratio (Moody & Saffell 1998) — maintain exponential moving averages of returns and squared returns, reward is the per-step change in estimated Sharpe
Transaction costs: cost = 0.001 * sum(|w_new - w_old|) applied every step
Reset: initialise to equal weight, start at a random point in the training window with a burn-in period long enough for rolling features to be valid
Include assertions to catch look-ahead bias and NaN propagation

Training (train.py)

Use stable-baselines3 PPO with MlpPolicy
Tune on validation set across: learning rate (3e-4, 1e-4), n_steps (2048, 4096), ent_coef (0.01)
Log to TensorBoard: episode reward, portfolio Sharpe, turnover per episode
Save checkpoints every 50k steps
Early stopping based on validation Sharpe

Baselines (baselines.py)

Equal weight: rebalance to 20% each asset monthly
Momentum: allocate proportionally to best performing asset over last 12 months, rebalance monthly
Buy and hold SPY
Shared evaluation function returning: annualised return, annualised volatility, Sharpe ratio, maximum drawdown, Calmar ratio

Evaluation (evaluate.py)

Load best checkpoint, run on held-out test set only once
Output a metrics table comparing PPO agent vs all three baselines
Plots: portfolio value over time, 252-day rolling Sharpe, weight allocation heatmap over time
Bootstrap confidence intervals on Sharpe ratio difference vs equal weight baseline

Ablation study (ablation.py)

Train three degraded variants and compare on validation set:

No transaction costs
No volatility features in state space
Raw returns reward instead of differential Sharpe


Output a comparison table showing the contribution of each design choice

Dependencies

yfinance, gymnasium, stable-baselines3, torch, numpy, pandas, matplotlib, seaborn, tensorboard
Include a requirements.txt

General

Clean, well-commented code throughout — I need to understand every component deeply for interview purposes
Add a README.md explaining the MDP formulation (state, action, transition, reward), the rationale for the differential Sharpe reward over alternatives, and a section on limitations and failure modes
Do not use any data from the validation or test sets during training or feature normalisation