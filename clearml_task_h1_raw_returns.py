"""
clearml_task_h1_raw_returns.py — ClearML task init for H1: raw returns in reward.

Run this once before starting training to register the task and hyperparameters
in the ClearML dashboard.

Usage:
    .venv/bin/python clearml_task_h1_raw_returns.py
"""

from clearml import Task

HYPERPARAMS: dict[str, float | int] = {
    "lr": 1e-4,
    "n_steps": 2048,
    "ent_coef": 0.01,
    "total_timesteps": 500_000,
    "window": 20,
    "transaction_cost": 0.001,
}

task = Task.init(
    project_name="rl-portfolio-agent",
    task_name="ppo-2026-03-08-h1-raw-returns-in-reward",
    reuse_last_task_id=False,
)
task.connect(HYPERPARAMS, name="hyperparameters")

print(f"ClearML task initialised: {task.id}")
print("Status: Running — start training now.")
