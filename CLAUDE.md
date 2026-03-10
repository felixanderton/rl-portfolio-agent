# CLAUDE.md

## Project

**Name**: rl-portfolio-agent
**Description**: Deep reinforcement learning portfolio allocation agent. Uses PPO (stable-baselines3) to dynamically allocate across 5 sector ETFs (XLK, XLE, XLF, XLV, XLI). Features a custom Gymnasium environment with a differential Sharpe ratio reward, transaction costs, and a state vector of log returns, rolling volatility, and mean-reversion signals. Includes comprehensive baselines, evaluation, and ablation study.
**Stack**: Python, yfinance, gymnasium, stable-baselines3, PyTorch, numpy, pandas, matplotlib, seaborn, tensorboard
**Repo**: https://github.com/felixanderton/rl-portfolio-agent
**Project board**: https://github.com/users/felixanderton/projects/3

## Agents

| Agent | Purpose |
|---|---|
| `reviewer` | Read-only code review — run before committing |
| `feature-writer` | Implements features from a spec |
| `test-writer` | Writes tests matching project conventions |
| `issue-writer` | Files GitHub issues for bugs and features |

## Workflow

1. Pick a deliverable from the project board
2. Create a worktree: `claude --worktree <feature-name>`
3. Invoke `feature-writer` with the issue description
4. Invoke `test-writer` to add coverage
5. Invoke `reviewer` before committing
6. Open a PR — one worktree per branch per PR

## Conventions

- Make the minimum change needed; do not refactor beyond the task scope
- Read existing code before modifying anything
- Never hardcode secrets — use environment variables
- Run `reviewer` before committing non-trivial changes

## Directory structure

```
src/        Python source (data.py, environment.py, train.py, evaluate.py, ablation.py, baselines.py, modal_train.py)
docs/       Experiment tracking and research (EXPERIMENT_LOG.md, HYPOTHESES.md, METRICS_GUIDE.md, ARCHITECTURE.md)
papers/     Reference PDFs
runs/       Gitignored training outputs (checkpoints/, logs/, best_model/, plots/)
```

Run training: `python src/train.py` from project root, or `.venv/bin/modal run src/modal_train.py` for Modal.

## ML Tracking

- ClearML project name: `rl-portfolio-agent`
- Invoke `ml-tracker` agent before and after each training run
- All experiments must have an entry in `docs/EXPERIMENT_LOG.md`
