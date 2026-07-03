#!/usr/bin/env bash
# rev6b: Transformer V0 (no PE, d=6, n_heads=2, d_ff=18) FL=50 with last-token loss.
# Usage: bash bash_scripts/train_attn_ltl.sh <gpu_id>

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh
conda activate LLMDPD

devices="${1:-0}"

echo "==== Transformer V0 FL=50 paper + LTL (GPU=${devices}) ===="
python main.py --step train_dpd \
  --dataset_name DPA_200MHz \
  --accelerator cuda --devices "${devices}" \
  --frame_length 50 --frame_stride 1 \
  --batch_size 64 --batch_size_eval 256 \
  --n_epochs 100 --opt_type adamw \
  --lr 1e-3 --lr_end 1e-6 --lr_schedule 1 --decay_factor 0.5 --patience 10 \
  --seed 0 \
  --PA_backbone dgru --PA_hidden_size 8 --PA_num_layers 1 \
  --DPD_backbone transformer --DPD_hidden_size 6 --DPD_num_layers 1 \
  --n_heads 2 --d_ff 18 --use_pos_encoding 0 \
  --last_token_loss 1 \
  --eval_match_train_len 0
echo "==== Transformer V0 LTL done ===="
