#!/usr/bin/env bash
# DPD on GaN_combined through frozen universal PA. LTL.
# Usage: bash train_combined_dpd.sh <arch: dgru|transformer|mamba> <FL> <gpu> [stride] [epochs]
set -euo pipefail
cd /home/mkiuyh/workspace/LLM-basedDPD/OpenDPD
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh && conda activate LLMDPD

ARCH="${1:?arch}"; FL="${2:?FL}"; devices="${3:-0}"; STRIDE="${4:-1}"; EP="${5:-100}"

if   [ "$FL" -le 100 ]; then BS=64
elif [ "$FL" -le 200 ]; then BS=64
elif [ "$FL" -le 500 ]; then BS=32
else BS=16; fi

# frozen universal PA (DGRU H=8) — symlink per FL
PA_SRC="save/GaN_combined/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt"
PA_DST="save/GaN_combined/train_pa/PA_S_0_M_DGRU_H_8_F_${FL}_P_486.pt"
[ "$FL" != "50" ] && [ ! -e "$PA_DST" ] && ln -sf "$(basename "$PA_SRC")" "$PA_DST"

case "$ARCH" in
  dgru)        EXTRA="--DPD_backbone dgru --DPD_hidden_size 8 --DPD_num_layers 1";;
  transformer) EXTRA="--DPD_backbone transformer --DPD_hidden_size 6 --DPD_num_layers 1 --n_heads 2 --d_ff 18 --use_pos_encoding 0";;
  mamba)       EXTRA="--DPD_backbone mamba --DPD_hidden_size 6 --DPD_num_layers 1 --mamba_d_state 4 --mamba_d_conv 4 --mamba_expand 2";;
  *) echo "unknown arch $ARCH"; exit 1;;
esac

echo "==== combined DPD ${ARCH} FL=${FL} stride=${STRIDE} ep=${EP} GPU=${devices} BS=${BS} ===="
python main.py --step train_dpd --dataset_name GaN_combined \
  --accelerator cuda --devices "${devices}" \
  --frame_length "${FL}" --frame_stride "${STRIDE}" \
  --batch_size "${BS}" --batch_size_eval 256 \
  --n_epochs "${EP}" --opt_type adamw \
  --lr 1e-3 --lr_end 1e-6 --lr_schedule 1 --decay_factor 0.5 --patience 10 \
  --seed 0 \
  --PA_backbone dgru --PA_hidden_size 8 --PA_num_layers 1 \
  ${EXTRA} --last_token_loss 1 --eval_match_train_len 0
echo "==== combined DPD ${ARCH} FL=${FL} done ===="
