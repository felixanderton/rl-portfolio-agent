# Architecture

## Overview

rl-portfolio-agent is a deep reinforcement learning system for dynamic portfolio allocation across five sector ETFs (XLK, XLE, XLF, XLV, XLI). A PPO agent trained with stable-baselines3 observes a state vector of log returns, rolling volatility, and mean-reversion signals, then outputs a continuous portfolio weight vector optimised via a differential Sharpe ratio reward that penalises transaction costs. The project is structured as a self-contained research pipeline: data ingestion, custom Gymnasium environment, competitive baselines, training, evaluation, and an ablation study isolating the contribution of each design choice.

## Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Framework / Runtime | stable-baselines3 (PPO), PyTorch |
| Environment | Gymnasium (custom env) |
| Data | yfinance, numpy, pandas |
| Visualisation | matplotlib, seaborn, tensorboard |
| Persistence | TBD |
| Infrastructure | TBD |

## Project Structure

```
rl-portfolio-agent/
  .claude/          # Claude Code config (agents, rules, settings)
  CLAUDE.md         # Claude Code instructions
  ARCHITECTURE.md   # This file
  data.py           # Data download, feature engineering, train/val/test splits
  environment.py    # Custom Gymnasium env with differential Sharpe reward
  baselines.py      # Equal-weight, momentum, buy-and-hold SPY baselines
  train.py          # PPO training loop, hyperparameter search, checkpointing
  evaluate.py       # Test-set evaluation, metrics table, plots, bootstrap CIs
  ablation.py       # Degraded variant training and comparison table
  README.md         # MDP formulation, design rationale, limitations
```

## Key Components

<!-- Fill in as components are built. -->

## Data Flow

Raw OHLCV data is downloaded via yfinance for XLK, XLE, XLF, XLV, XLI and SPY. Adjusted close prices are used to compute log returns, rolling volatility (21-day), and mean-reversion z-scores. The resulting feature matrix is normalised and split into train (2000-2015), val (2015-2019), and test (2020-2024) numpy arrays. The Gymnasium environment wraps the train array; at each step the agent receives a state vector, emits a 5-dimensional weight vector (softmax-normalised), and receives a differential Sharpe ratio reward net of transaction costs. The trained PPO policy is evaluated once on the held-out test set and compared against all baselines.

## Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-03-08 | Chose PPO via stable-baselines3 | Well-tuned off-the-shelf implementation; on-policy gradient well-suited to non-stationary financial time series |
| 2026-03-08 | Differential Sharpe reward | Optimises risk-adjusted return directly rather than raw PnL; differentiable approximation avoids episode-length reward sparsity |
| 2026-03-08 | Transaction cost penalty (0.001 × sum\|Δw\|) | Prevents degenerate high-turnover policies; realistic proxy for ETF bid-ask spread |
