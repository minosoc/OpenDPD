#!/usr/bin/env bash
# Retrain DGRU H=8 with paper config (eval_match_train_len=0, nperseg=2560 eval) at given FL.
# Usage: bash bash_scripts/retrain_dgru_fl_paper.sh <FL> <gpu_id>

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh
conda activate LLMDPD

FL="${1:?usage: $0 <FL> <gpu_id>}"
devices="${2:-0}"

# PA reuse — symlink to FL=50 PA (DGRU is length-invariant; same paper-trained PA OK)
PA_SRC="save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt"
PA_DST="save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_${FL}_P_486.pt"
if [ ! -f "${PA_DST}" ] && [ "${FL}" != "50" ]; then
    ln -sf "$(basename ${PA_SRC})" "${PA_DST}"
fi

# Adjust batch for memory
if   [ "${FL}" -le 200 ];  then BS=64
elif [ "${FL}" -le 500 ];  then BS=64
elif [ "${FL}" -le 1000 ]; then BS=32
else                            BS=16
fi

echo "==== DGRU FL=${FL} paper-config retrain on GPU ${devices} (BS=${BS}) ===="
python main.py --step train_dpd \
  --dataset_name DPA_200MHz \
  --accelerator cuda --devices "${devices}" \
  --frame_length "${FL}" --frame_stride 1 \
  --batch_size "${BS}" --batch_size_eval 256 \
  --n_epochs 100 --opt_type adamw \
  --lr 1e-3 --lr_end 1e-6 --lr_schedule 1 --decay_factor 0.5 --patience 10 \
  --seed 0 \
  --PA_backbone dgru --PA_hidden_size 8 --PA_num_layers 1 \
  --DPD_backbone dgru --DPD_hidden_size 8 --DPD_num_layers 1 \
  --eval_match_train_len 0
echo "==== DGRU FL=${FL} done ===="
