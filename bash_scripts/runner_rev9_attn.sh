#!/usr/bin/env bash
# rev9: Transformer V0 lookahead sweep at FL=201. M ∈ {0,5,20,50,100}.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
devices="${1:-0}"
for M in 0 5 20 50 100; do
    bash "${SCRIPT_DIR}/train_lookahead.sh" transformer "${M}" "${devices}" \
        || echo "  [WARN] transformer M=${M} failed"
done
echo "==== rev9 transformer sweep done (GPU=${devices}) ===="
