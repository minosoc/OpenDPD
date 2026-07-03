#!/usr/bin/env bash
# rev10d: k=11, 41 on one GPU.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
devices="${1:-0}"
for K in 11 41; do
    bash "${SCRIPT_DIR}/train_rev10d.sh" "${K}" "${devices}" \
        || echo "  [WARN] k=${K} failed"
done
echo "==== rev10d a-runner done (GPU=${devices}) ===="
