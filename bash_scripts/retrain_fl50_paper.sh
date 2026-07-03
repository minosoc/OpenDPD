#!/usr/bin/env bash
# Retrain three FL=50 DPDs with PAPER config (no eval_match_train_len, nperseg=2560 eval).
#
# Saves to save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_50/ with PE-tagged filenames.
# PA is reused from existing paper-trained PA (DGRU H=8 F=50).
#
# Usage: bash bash_scripts/retrain_fl50_paper.sh <backbone_tag> <gpu_id>

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh
conda activate LLMDPD

bb_tag="${1:?usage: $0 <backbone_tag> <gpu_id>}"
devices="${2:-0}"

common=(
  --dataset_name DPA_200MHz
  --accelerator cuda
  --devices "${devices}"
  --frame_length 50
  --frame_stride 1
  --batch_size 64
  --batch_size_eval 256
  --n_epochs 100
  --opt_type adamw
  --lr 1e-3
  --lr_end 1e-6
  --lr_schedule 1
  --decay_factor 0.5
  --patience 10
  --seed 0
  --PA_backbone dgru
  --PA_hidden_size 8
  --PA_num_layers 1
  --eval_match_train_len 0
)

case "${bb_tag}" in
  dgru)
    dpd=(--DPD_backbone dgru --DPD_hidden_size 8 --DPD_num_layers 1)
    ;;
  attn)
    dpd=(--DPD_backbone transformer --DPD_hidden_size 6 --DPD_num_layers 1
         --n_heads 2 --d_ff 18 --use_pos_encoding 0)
    ;;
  attn_pe)
    dpd=(--DPD_backbone transformer --DPD_hidden_size 6 --DPD_num_layers 1
         --n_heads 2 --d_ff 18 --use_pos_encoding 1)
    ;;
esac

echo "==== Retraining ${bb_tag} FL=50 (paper config) on GPU ${devices} ===="
python main.py --step train_dpd "${common[@]}" "${dpd[@]}"
echo "==== ${bb_tag} done ===="
