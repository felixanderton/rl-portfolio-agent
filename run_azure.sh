#!/usr/bin/env bash
# run_azure.sh — Launch a training run on an Azure spot VM.
#
# The VM clones the GitHub repo and runs train.py directly — no local Docker
# build required. The DSVM image has Python, CUDA, and PyTorch pre-installed.
#
# Usage:
#   ./run_azure.sh
#
# Prerequisites:
#   - az login completed
#   - Latest code pushed to GitHub (git push origin main)
#   - .env filled in

set -euo pipefail

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
if [[ ! -f .env ]]; then
    echo "ERROR: .env not found. Fill in credentials before running." >&2
    exit 1
fi
source .env

: "${CLEARML_API_HOST:?CLEARML_API_HOST is not set in .env}"
: "${CLEARML_API_ACCESS_KEY:?CLEARML_API_ACCESS_KEY is not set in .env}"
: "${CLEARML_API_SECRET_KEY:?CLEARML_API_SECRET_KEY is not set in .env}"
: "${AZURE_RESOURCE_GROUP:?}"
: "${AZURE_LOCATION:?}"
: "${AZURE_VM_SIZE:?}"
: "${AZURE_VM_NAME:?}"

REPO_URL="https://github.com/felixanderton/rl-portfolio-agent"

# ---------------------------------------------------------------------------
# 1. Resource group
# ---------------------------------------------------------------------------
echo "=== [1/3] Resource group ==="
az group create \
    --name "$AZURE_RESOURCE_GROUP" \
    --location "$AZURE_LOCATION" \
    --output none
echo "    OK: $AZURE_RESOURCE_GROUP"

# ---------------------------------------------------------------------------
# 2. Generate startup script
#    ClearML credentials are injected here on the local machine.
# ---------------------------------------------------------------------------
echo "=== [2/3] Generating startup script ==="
STARTUP=$(mktemp /tmp/rl-startup-XXXX.sh)
cat > "$STARTUP" <<SCRIPT
#!/bin/bash
set -euo pipefail
exec > /var/log/rl-training.log 2>&1

echo "--- Cloning repo ---"
git clone ${REPO_URL} /app
cd /app

echo "--- Installing dependencies ---"
pip install -r requirements.txt

echo "--- Starting training ---"
export CLEARML_API_HOST='${CLEARML_API_HOST}'
export CLEARML_API_ACCESS_KEY='${CLEARML_API_ACCESS_KEY}'
export CLEARML_API_SECRET_KEY='${CLEARML_API_SECRET_KEY}'
python train.py

echo "--- Training complete. Powering off. ---"
poweroff
SCRIPT
chmod +x "$STARTUP"
echo "    OK"

# ---------------------------------------------------------------------------
# 3. Accept marketplace terms and launch spot VM
# ---------------------------------------------------------------------------
echo "=== [3/3] Launching spot VM ==="
az vm image terms accept \
    --publisher microsoft-dsvm \
    --offer ubuntu-2004 \
    --plan 2004 \
    --output none 2>/dev/null || true

az vm create \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --name "$AZURE_VM_NAME" \
    --image "microsoft-dsvm:ubuntu-2004:2004:latest" \
    --plan-name "2004" \
    --plan-publisher "microsoft-dsvm" \
    --plan-product "ubuntu-2004" \
    --size "$AZURE_VM_SIZE" \
    --priority Spot \
    --eviction-policy Deallocate \
    --custom-data "@$STARTUP" \
    --generate-ssh-keys \
    --output table

rm -f "$STARTUP"

VM_IP=$(az vm show \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --name "$AZURE_VM_NAME" \
    --show-details \
    --query publicIps \
    --output tsv)

echo ""
echo "================================================================"
echo "  VM launched: $VM_IP"
echo "  Monitor:     ssh $VM_IP 'tail -f /var/log/rl-training.log'"
echo "  Results:     ClearML dashboard (run logs + model artifact)"
echo "  Tear down:   az vm delete -g $AZURE_RESOURCE_GROUP -n $AZURE_VM_NAME --yes --no-wait"
echo "================================================================"
