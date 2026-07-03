#!/usr/bin/env bash
# rev9: Lookahead-M DPD train at fixed FL_total=201. backbone ∈ {dgru, transformer, mamba}.
# Usage: bash bash_scripts/train_lookahead.sh <backbone> <M> <gpu_id>

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh
conda activate LLMDPD

BACKBONE="${1:?usage: $0 <backbone> <M> <gpu_id>}"
M="${2:?usage: $0 <backbone> <M> <gpu_id>}"
devices="${3:-0}"
FL=201
BS=64

PA_SRC="save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt"
PA_DST="save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_${FL}_P_486.pt"
[ ! -f "${PA_DST}" ] && ln -sf "$(basename ${PA_SRC})" "${PA_DST}"

case "${BACKBONE}" in
  dgru)
    BB_ARGS="--DPD_backbone dgru --DPD_hidden_size 8 --DPD_num_layers 1"
    ;;
  transformer)
    BB_ARGS="--DPD_backbone transformer --DPD_hidden_size 6 --DPD_num_layers 1 --n_heads 2 --d_ff 18 --use_pos_encoding 0"
    ;;
  mamba)
    BB_ARGS="--DPD_backbone mamba --DPD_hidden_size 6 --DPD_num_layers 1 --mamba_d_state 4 --mamba_d_conv 4 --mamba_expand 2"
    ;;
  *)
    echo "Unknown backbone: ${BACKBONE}"; exit 1
    ;;
esac

echo "==== ${BACKBONE} FL=${FL} M=${M} GPU=${devices} ===="
python main.py --step train_dpd \
  --dataset_name DPA_200MHz \
  --accelerator cuda --devices "${devices}" \
  --frame_length "${FL}" --frame_stride 1 \
  --batch_size "${BS}" --batch_size_eval 256 \
  --n_epochs 100 --opt_type adamw \
  --lr 1e-3 --lr_end 1e-6 --lr_schedule 1 --decay_factor 0.5 --patience 10 \
  --seed 0 \
  --PA_backbone dgru --PA_hidden_size 8 --PA_num_layers 1 \
  ${BB_ARGS} \
  --last_token_loss 1 --lookahead "${M}" \
  --eval_match_train_len 0
echo "==== ${BACKBONE} FL=${FL} M=${M} done ===="
