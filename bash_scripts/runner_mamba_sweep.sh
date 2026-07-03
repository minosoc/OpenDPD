#!/usr/bin/env bash
# Sequentially train Mamba at FL=50,100,200,500,1000 on one GPU.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
devices="${1:-0}"
for FL in 50 100 200 500 1000; do
    echo "==== Mamba FL=${FL} on GPU=${devices} ===="
    bash "${SCRIPT_DIR}/train_mamba.sh" "${FL}" "${devices}" \
        || echo "  [WARN] run failed for FL=${FL}, continuing"
done
echo "==== Mamba sweep done on GPU=${devices} ===="
