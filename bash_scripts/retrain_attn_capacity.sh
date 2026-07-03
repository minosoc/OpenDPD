#!/usr/bin/env bash
# Transformer no-PE capacity scaling: d_model=16 (2898 params) and d_model=32 (10914 params).
# Usage: bash bash_scripts/retrain_attn_capacity.sh <d_model> <FL> <gpu_id>

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh
conda activate LLMDPD

DM="${1:?usage: $0 <d_model> <FL> <gpu_id>}"
FL="${2:?usage: $0 <d_model> <FL> <gpu_id>}"
devices="${3:-0}"

DFF=$((3 * DM))

PA_SRC="save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt"
PA_DST="save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_${FL}_P_486.pt"
[ ! -f "${PA_DST}" ] && [ "${FL}" != "50" ] && ln -sf "$(basename ${PA_SRC})" "${PA_DST}"

# Batch size scaling — attention is O(T^2 * d_model)
if   [ "${FL}" -le 100 ];  then BS=64
elif [ "${FL}" -le 200 ];  then BS=64
elif [ "${FL}" -le 500 ];  then BS=32
elif [ "${FL}" -le 1000 ]; then BS=16
else                            BS=8
fi
# Larger d_model also pushes memory
if [ "${DM}" -ge 32 ] && [ "${FL}" -ge 500 ]; then BS=$((BS / 2)); fi

echo "==== Transformer noPE d_model=${DM} d_ff=${DFF} FL=${FL} GPU=${devices} (BS=${BS}) ===="
python main.py --step train_dpd \
  --dataset_name DPA_200MHz \
  --accelerator cuda --devices "${devices}" \
  --frame_length "${FL}" --frame_stride 1 \
  --batch_size "${BS}" --batch_size_eval 256 \
  --n_epochs 100 --opt_type adamw \
  --lr 1e-3 --lr_end 1e-6 --lr_schedule 1 --decay_factor 0.5 --patience 10 \
  --seed 0 \
  --PA_backbone dgru --PA_hidden_size 8 --PA_num_layers 1 \
  --DPD_backbone transformer --DPD_hidden_size "${DM}" --DPD_num_layers 1 \
  --n_heads 2 --d_ff "${DFF}" --use_pos_encoding 0 \
  --eval_match_train_len 0
echo "==== d_model=${DM} FL=${FL} done ===="
