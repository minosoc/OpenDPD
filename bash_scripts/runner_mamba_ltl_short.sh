#!/usr/bin/env bash
# rev7b: FL=50,100,200 sequential.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
devices="${1:-0}"
for FL in 50 100 200; do
    bash "${SCRIPT_DIR}/train_mamba_ltl.sh" "${FL}" "${devices}" \
        || echo "  [WARN] FL=${FL} failed"
done
echo "==== Mamba LTL short sweep done (GPU=${devices}) ===="
