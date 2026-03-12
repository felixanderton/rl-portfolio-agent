"""
plot_model.py — Run diagnostic plots on a saved model without retraining.

Loads a model from a local path, then generates the same event-zoom plots
(training and validation) that train.py produces at the end of a run.

Usage:
    python src/plot_model.py <path/to/model>   # .zip extension optional

Example:
    python src/plot_model.py runs/best_model/best_model
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from clearml import Task
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import TICKERS, load_data
from train import (
    CLEARML_PROJECT,
    _TRAIN_EVENTS,
    _VAL_EVENTS,
    _plot_event_zoom,
    _rollout,
    run_validation,
)


def main(model_path: str) -> None:

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading data ...")
    data = load_data()
    train_features = data["train"]
    train_prices = data["train_prices"]
    train_dates: pd.DatetimeIndex = data["train_dates"]
    val_features = data["val"]
    val_prices = data["val_prices"]
    val_dates: pd.DatetimeIndex = data["val_dates"]

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {model_path} ...")
    model = PPO.load(model_path)

    # ------------------------------------------------------------------
    # 3. Create a ClearML diagnostic task to hold the plots
    # ------------------------------------------------------------------
    model_name = Path(model_path).stem
    task = Task.init(
        project_name=CLEARML_PROJECT,
        task_name=f"diagnostic-plots/{model_name}",
        reuse_last_task_id=False,
    )
    task.connect({"model_path": model_path}, name="source")

    # ------------------------------------------------------------------
    # 4. Validation plots (reuses full run_validation)
    # ------------------------------------------------------------------
    print("Running validation rollout ...")
    run_validation(model, val_features, val_prices, val_dates, task)

    # ------------------------------------------------------------------
    # 5. Training event zoom
    # ------------------------------------------------------------------
    print("Running training rollout ...")
    window = 20
    train_weights_arr, train_prices_arr = _rollout(model, train_features, train_prices)
    train_episode_dates: pd.DatetimeIndex = train_dates[
        window : window + len(train_weights_arr)
    ]
    final_step = len(train_weights_arr)
    _plot_event_zoom(
        train_weights_arr,
        train_prices_arr,
        train_episode_dates,
        TICKERS,
        task,
        final_step,
        events=_TRAIN_EVENTS,
        title="train_event_zoom",
    )

    task.close()
    print(f"\nPlots uploaded to ClearML task: {task.id}")
    print(f"View at: {task.get_output_log_web_page()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path/to/model>")
        sys.exit(1)
    main(sys.argv[1])
