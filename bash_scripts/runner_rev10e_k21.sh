#!/usr/bin/env bash
# rev10e: k=21 with lr ∈ {3e-4, 1e-4}.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
devices="${1:-0}"
for LR in 3e-4 1e-4; do
    bash "${SCRIPT_DIR}/train_rev10e.sh" 21 "${LR}" "${devices}" \
        || echo "  [WARN] k=21 lr=${LR} failed"
done
echo "==== rev10e k=21 runner done (GPU=${devices}) ===="
