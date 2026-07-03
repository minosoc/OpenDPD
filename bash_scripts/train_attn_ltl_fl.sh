#!/usr/bin/env bash
# rev8: Transformer V0 (no PE, d=6, n_heads=2, d_ff=18) LTL at given FL.
# Usage: bash bash_scripts/train_attn_ltl_fl.sh <FL> <gpu_id>

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh
conda activate LLMDPD

FL="${1:?usage: $0 <FL> <gpu_id>}"
devices="${2:-0}"

PA_SRC="save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt"
PA_DST="save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_${FL}_P_486.pt"
[ ! -f "${PA_DST}" ] && [ "${FL}" != "50" ] && ln -sf "$(basename ${PA_SRC})" "${PA_DST}"

if   [ "${FL}" -le 100 ];  then BS=64
elif [ "${FL}" -le 200 ];  then BS=64
elif [ "${FL}" -le 500 ];  then BS=32
elif [ "${FL}" -le 1000 ]; then BS=16
else                            BS=8
fi

echo "==== Transformer V0 LTL d=6 FL=${FL} GPU=${devices} (BS=${BS}) ===="
python main.py --step train_dpd \
  --dataset_name DPA_200MHz \
  --accelerator cuda --devices "${devices}" \
  --frame_length "${FL}" --frame_stride 1 \
  --batch_size "${BS}" --batch_size_eval 256 \
  --n_epochs 100 --opt_type adamw \
  --lr 1e-3 --lr_end 1e-6 --lr_schedule 1 --decay_factor 0.5 --patience 10 \
  --seed 0 \
  --PA_backbone dgru --PA_hidden_size 8 --PA_num_layers 1 \
  --DPD_backbone transformer --DPD_hidden_size 6 --DPD_num_layers 1 \
  --n_heads 2 --d_ff 18 --use_pos_encoding 0 \
  --last_token_loss 1 \
  --eval_match_train_len 0
echo "==== Transformer V0 LTL FL=${FL} done ===="
