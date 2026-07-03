#!/usr/bin/env bash
# rev5: Transformer 4 advanced variants at FL=50 paper config.
# Usage: bash bash_scripts/train_attn_rev5.sh <d|e|f|g> <gpu_id>

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh
conda activate LLMDPD

V="${1:?usage: $0 <d|e|f|g> <gpu_id>}"
devices="${2:-0}"

case "${V}" in
  d)  NH=2; NL=1; CSK=3;  LAW=0 ;;  # Conv1d stem k=3
  e)  NH=2; NL=1; CSK=0;  LAW=7 ;;  # Local attention window=7
  f)  NH=2; NL=2; CSK=0;  LAW=0 ;;  # 2 layers
  g)  NH=1; NL=1; CSK=0;  LAW=0 ;;  # n_heads=1, head_dim=6
  *) echo "Unknown variant $V"; exit 1 ;;
esac

echo "==== rev5_${V}: NH=${NH} NL=${NL} CSK=${CSK} LAW=${LAW} FL=50 GPU=${devices} ===="
python main.py --step train_dpd \
  --dataset_name DPA_200MHz \
  --accelerator cuda --devices "${devices}" \
  --frame_length 50 --frame_stride 1 \
  --batch_size 64 --batch_size_eval 256 \
  --n_epochs 100 --opt_type adamw \
  --lr 1e-3 --lr_end 1e-6 --lr_schedule 1 --decay_factor 0.5 --patience 10 \
  --seed 0 \
  --PA_backbone dgru --PA_hidden_size 8 --PA_num_layers 1 \
  --DPD_backbone transformer --DPD_hidden_size 6 --DPD_num_layers "${NL}" \
  --n_heads "${NH}" --d_ff 18 --use_pos_encoding 0 \
  --conv_stem_kernel "${CSK}" --local_attn_window "${LAW}" \
  --eval_match_train_len 0
echo "==== rev5_${V} done ===="
