#!/usr/bin/env bash
# Run all frame_length jobs for one backbone on one GPU sequentially.
# Usage: bash bash_scripts/sweep_runner.sh <backbone_tag> <gpu_id>

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bb_tag="${1:?usage: $0 <backbone_tag> <gpu_id>}"
devices="${2:-0}"

LENGTHS=(50 100 500 1000 2000 3000 4000 5000 10000)

for FL in "${LENGTHS[@]}"; do
    echo "=========================================="
    echo "[${bb_tag}/FL=${FL}/GPU=${devices}] starting"
    echo "=========================================="
    bash "${SCRIPT_DIR}/sweep_framelen.sh" "${bb_tag}" "${FL}" "${devices}" \
        || echo "  [WARN] run failed for ${bb_tag} FL=${FL}, continuing"
done

echo "=========================================="
echo "[${bb_tag}/GPU=${devices}] ALL DONE"
echo "=========================================="
