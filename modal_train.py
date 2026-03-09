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
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .add_local_file("requirements.txt", "/requirements.txt")
    .pip_install_from_requirements("/requirements.txt")
)

REPO_URL = "https://github.com/felixanderton/rl-portfolio-agent"


@app.function(
    image=image,
    cpu=16,
    memory=32768,
    timeout=7200,
    secrets=[modal.Secret.from_name("clearml-credentials")],
)
def train():
    subprocess.run(["git", "clone", REPO_URL, "/app"], check=True)
    os.chdir("/app")
    subprocess.run(["python", "train.py"], check=True)


@app.local_entrypoint()
def main():
    train.remote()
