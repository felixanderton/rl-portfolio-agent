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

import json
import logging
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any

import modal

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

app = modal.App("rl-portfolio-agent")

checkpoint_vol = modal.Volume.from_name("rl-checkpoints", create_if_missing=True)
VOL_MOUNT = "/vol/checkpoints"

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

# SIGTERM flag — set by the handler, read by _PersistenceCallback inside train()
_SIGTERM_RECEIVED = False


def _sigterm_handler(signum: int, frame: object) -> None:
    global _SIGTERM_RECEIVED
    _SIGTERM_RECEIVED = True


signal.signal(signal.SIGTERM, _sigterm_handler)

# ---------------------------------------------------------------------------
# Volume helpers
# ---------------------------------------------------------------------------


def _vol_dir(run_name: str) -> Path:
    return Path(VOL_MOUNT) / run_name


def _read_meta(run_name: str) -> dict[str, Any] | None:
    p = _vol_dir(run_name) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def _write_meta(
    run_name: str, task_id: str, best_sharpe: float, steps_done: int
) -> None:
    meta = {
        "clearml_task_id": task_id,
        "best_sharpe": best_sharpe,
        "steps_done": steps_done,
    }
    d = _vol_dir(run_name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(meta))
    checkpoint_vol.commit()  # must commit — writes are not durable without this


@app.function(
    image=image,
    cpu=16,
    memory=32768,
    timeout=7200,
    secrets=[modal.Secret.from_name("clearml-credentials")],
    volumes={VOL_MOUNT: checkpoint_vol},
)
def train() -> None:
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, "/app"], check=True)
    os.chdir("/app/src")

    import sys

    sys.path.insert(0, "/app/src")

    from train import (  # noqa: PLC0415
        CHECKPOINT_FREQ,
        CHECKPOINT_DIR,
        ENT_COEF,
        LEARNING_RATE,
        N_STEPS,
        RunConfig,
        PeriodicValCallback,
        main,
    )
    from stable_baselines3.common.callbacks import BaseCallback  # noqa: PLC0415

    # ------------------------------------------------------------------
    # Persistence callback — defined here to close over train-module symbols
    # ------------------------------------------------------------------
    class _PersistenceCallback(BaseCallback):
        def __init__(
            self,
            run_name: str,
            task_id_out: list[str],
            val_cb_out: list[PeriodicValCallback],
        ) -> None:
            super().__init__()
            self._run_name = run_name
            self._task_id_out = task_id_out
            self._val_cb_out = val_cb_out

        def _on_step(self) -> bool:
            if _SIGTERM_RECEIVED:
                return False
            if (
                self.num_timesteps % CHECKPOINT_FREQ == 0
                and self._task_id_out
                and self._val_cb_out
            ):
                src = (
                    CHECKPOINT_DIR
                    / self._run_name
                    / f"ppo_checkpoint_{self.num_timesteps}_steps.zip"
                )
                if src.exists():
                    vol_dir = _vol_dir(self._run_name)
                    vol_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src, vol_dir / "latest.zip")
                    _write_meta(
                        self._run_name,
                        self._task_id_out[0],
                        self._val_cb_out[0]._best_sharpe,
                        self.num_timesteps,
                    )
                    logger.info(
                        f"Checkpoint persisted to volume at step {self.num_timesteps}"
                    )
            return True

    # ------------------------------------------------------------------
    # Determine run_name (needed to look up volume checkpoint)
    # ------------------------------------------------------------------
    cfg = RunConfig.model_validate(
        {"learning_rate": LEARNING_RATE, "n_steps": N_STEPS, "ent_coef": ENT_COEF}
    )
    run_name = cfg.run_name

    # ------------------------------------------------------------------
    # Resume detection
    # ------------------------------------------------------------------
    meta = _read_meta(run_name)
    if meta is not None:
        import train as train_mod  # noqa: PLC0415

        train_mod.WARM_START_PATH = str(_vol_dir(run_name) / "latest.zip")
        resume_steps: int = meta["steps_done"]
        resume_best_sharpe: float = meta["best_sharpe"]
        clearml_task_id: str | None = meta["clearml_task_id"]
        logger.info(f"Resuming from step {resume_steps} (task {clearml_task_id})")
    else:
        resume_steps = 0
        resume_best_sharpe = -float("inf")
        clearml_task_id = None

        # Download warm-start model from ClearML
        from clearml import Task as ClearMLTask  # noqa: PLC0415

        warm_start_dir = Path(WARM_START_DIR)
        warm_start_dir.mkdir(parents=True, exist_ok=True)
        warm_start_task: ClearMLTask = ClearMLTask.get_task(task_id=WARM_START_TASK_ID)
        artifact_path = warm_start_task.artifacts[WARM_START_ARTIFACT].get_local_copy()
        shutil.copy(artifact_path, warm_start_dir / "best_model.zip")

        import train as train_mod  # noqa: PLC0415

        train_mod.WARM_START_PATH = str(warm_start_dir / "best_model.zip")

    # ------------------------------------------------------------------
    # Build out-lists and persistence callback
    # ------------------------------------------------------------------
    task_id_out: list[str] = []
    val_cb_out: list[PeriodicValCallback] = []
    persistence_cb = _PersistenceCallback(run_name, task_id_out, val_cb_out)

    # ------------------------------------------------------------------
    # Train (and clean up volume on success)
    # ------------------------------------------------------------------
    try:
        main(
            resume_steps=resume_steps,
            resume_best_sharpe=resume_best_sharpe,
            clearml_task_id=clearml_task_id,
            extra_callbacks=[persistence_cb],
            task_id_out=task_id_out,
            val_cb_out=val_cb_out,
        )
        shutil.rmtree(_vol_dir(run_name), ignore_errors=True)
        checkpoint_vol.commit()
    except Exception:
        logger.exception("Training failed — volume checkpoint preserved for retry")
        raise


@app.local_entrypoint()
def main() -> None:
    train.remote()
