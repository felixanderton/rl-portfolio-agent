"""
modal_train.py — Run train.py on Modal cloud compute.

Usage:
    modal run modal_train.py

Prerequisites:
    pip install modal
    modal token new
    modal secret create clearml-credentials \
        CLEARML_API_HOST="https://api.clear.ml" \
        CLEARML_API_ACCESS_KEY="..." \
        CLEARML_API_SECRET_KEY="..."

Note: push your latest code to GitHub before each run — the VM clones fresh.
"""

import subprocess
import os
import modal

app = modal.App("rl-portfolio-agent")

# Dependencies are installed at image build time (cached between runs).
# requirements.txt is copied in so the pip layer is invalidated only when
# dependencies change, not on every code edit.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "yfinance",
        "gymnasium",
        "stable-baselines3[extra]",
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "tensorboard",
        "tqdm",
        "clearml",
        "pydantic",
    )
)

REPO_URL = "https://github.com/felixanderton/rl-portfolio-agent"
BRANCH = "hypothesis/H13"

# ClearML task ID for the warm-start model (H6 best model, val Sharpe 0.7056)
# H10 has no uploaded artifact; H6 is the most recent available checkpoint.
WARM_START_TASK_ID = "40f1afcadac442e2b78a0b40f6f72f01"
WARM_START_ARTIFACT = "best_model"
WARM_START_DIR = "/app/runs/warm_start"


@app.function(
    image=image,
    cpu=16,
    memory=32768,
    timeout=7200,
    secrets=[modal.Secret.from_name("clearml-credentials")],
)
def train():
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, "/app"], check=True)
    os.chdir("/app/src")

    # Download warm-start model from ClearML before training
    import sys
    import shutil
    from pathlib import Path
    from clearml import Task as ClearMLTask

    warm_start_dir = Path(WARM_START_DIR)
    warm_start_dir.mkdir(parents=True, exist_ok=True)
    warm_start_task = ClearMLTask.get_task(task_id=WARM_START_TASK_ID)
    artifact_path = warm_start_task.artifacts[WARM_START_ARTIFACT].get_local_copy()
    shutil.copy(artifact_path, warm_start_dir / "best_model.zip")

    sys.path.insert(0, "/app/src")
    from train import main

    main()


@app.local_entrypoint()
def main():
    train.remote()
