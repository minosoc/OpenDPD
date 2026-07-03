#!/usr/bin/env bash
# Run all FL sweeps for one d_model on one GPU sequentially.
# Usage: bash bash_scripts/runner_attn_capacity.sh <d_model> <gpu_id>

set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DM="${1:?usage: $0 <d_model> <gpu_id>}"
devices="${2:-0}"

for FL in 50 100 200 500 1000; do
    echo "=========================================="
    echo "[d_model=${DM}/FL=${FL}/GPU=${devices}] starting"
    echo "=========================================="
    bash "${SCRIPT_DIR}/retrain_attn_capacity.sh" "${DM}" "${FL}" "${devices}" \
        || echo "  [WARN] run failed for d_model=${DM} FL=${FL}, continuing"
done
echo "=========================================="
echo "[d_model=${DM}/GPU=${devices}] ALL DONE"
echo "=========================================="
