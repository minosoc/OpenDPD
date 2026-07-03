#!/usr/bin/env bash
# rev8: Transformer V0 LTL FL=100,200,500,1000 sequential.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
devices="${1:-0}"
for FL in 100 200 500 1000; do
    bash "${SCRIPT_DIR}/train_attn_ltl_fl.sh" "${FL}" "${devices}" \
        || echo "  [WARN] Transformer LTL FL=${FL} failed"
done
echo "==== rev8 Transformer LTL sweep done (GPU=${devices}) ===="
