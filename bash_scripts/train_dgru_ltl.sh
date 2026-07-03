#!/usr/bin/env bash
# rev6: DGRU H=8 FL=50 with last-token loss (matches sliding-inference extraction).
# Usage: bash bash_scripts/train_dgru_ltl.sh <gpu_id>

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh
conda activate LLMDPD

devices="${1:-0}"

echo "==== DGRU H=8 FL=50 paper config + last-token loss (GPU=${devices}) ===="
python main.py --step train_dpd \
  --dataset_name DPA_200MHz \
  --accelerator cuda --devices "${devices}" \
  --frame_length 50 --frame_stride 1 \
  --batch_size 64 --batch_size_eval 256 \
  --n_epochs 100 --opt_type adamw \
  --lr 1e-3 --lr_end 1e-6 --lr_schedule 1 --decay_factor 0.5 --patience 10 \
  --seed 0 \
  --PA_backbone dgru --PA_hidden_size 8 --PA_num_layers 1 \
  --DPD_backbone dgru --DPD_hidden_size 8 --DPD_num_layers 1 \
  --last_token_loss 1 \
  --eval_match_train_len 0
echo "==== DGRU LTL done ===="
