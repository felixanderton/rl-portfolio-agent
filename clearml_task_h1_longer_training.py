"""
ClearML task initialisation for H1: longer training (500k -> 1.5M steps).

Run this script before launching train.py to register the experiment in
ClearML with the correct hyperparameters and task name.
"""

from clearml import Task

hyperparams = {
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "ent_coef": 0.01,
    "total_timesteps": 1_500_000,
    "window": 20,
    "transaction_cost": 0.001,
}

task = Task.init(
    project_name="rl-portfolio-agent",
    task_name="ppo-2026-03-08-longer-training-1.5M-steps",
    reuse_last_task_id=False,
)
task.connect(hyperparams, name="hyperparameters")
task.close()
