#!/usr/bin/env bash
# rev7b: Sequentially train Mamba LTL at FL=50,100,200,500,1000 on one GPU.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
devices="${1:-0}"
for FL in 50 100 200 500 1000; do
    echo "==== Mamba LTL FL=${FL} on GPU=${devices} ===="
    bash "${SCRIPT_DIR}/train_mamba_ltl.sh" "${FL}" "${devices}" \
        || echo "  [WARN] run failed for FL=${FL}, continuing"
done
echo "==== Mamba LTL sweep done on GPU=${devices} ===="
