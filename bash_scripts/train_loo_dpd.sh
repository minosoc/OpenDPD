#!/usr/bin/env bash
# LOO zero-shot DPD: train on pooled 3 conditions, universal PA frozen.
# Usage: train_loo_dpd.sh <held: c24|c27|b20|b24> <arch> <FL> <gpu> [stride]
set -uo pipefail
cd /home/mkiuyh/workspace/LLM-basedDPD/OpenDPD
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh && conda activate LLMDPD
HELD="${1:?}"; ARCH="${2:?}"; FL="${3:?}"; gpu="${4:-0}"; STRIDE="${5:-4}"
if   [ "$FL" -le 100 ]; then BS=64
elif [ "$FL" -le 200 ]; then BS=64
elif [ "$FL" -le 500 ]; then BS=32
else BS=16; fi
UNIV="$(pwd)/save/GaN_combined/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt"
PADIR="save/GaN_loo_${HELD}/train_pa"; mkdir -p "$PADIR"
ln -sf "$UNIV" "$PADIR/PA_S_0_M_DGRU_H_8_F_${FL}_P_486.pt"
case "$ARCH" in
  dgru)  EXTRA="--DPD_backbone dgru --DPD_hidden_size 8 --DPD_num_layers 1";;
  mamba) EXTRA="--DPD_backbone mamba --DPD_hidden_size 6 --DPD_num_layers 1 --mamba_d_state 4 --mamba_d_conv 4 --mamba_expand 2";;
esac
echo "==== LOO held=$HELD $ARCH FL=$FL GPU=$gpu ===="
python main.py --step train_dpd --dataset_name "GaN_loo_${HELD}" \
  --accelerator cuda --devices "$gpu" \
  --frame_length "$FL" --frame_stride "$STRIDE" --batch_size "$BS" --batch_size_eval 256 \
  --n_epochs 100 --opt_type adamw --lr 1e-3 --lr_end 1e-6 --lr_schedule 1 --decay_factor 0.5 --patience 10 \
  --seed 0 --PA_backbone dgru --PA_hidden_size 8 --PA_num_layers 1 \
  ${EXTRA} --last_token_loss 1 --eval_match_train_len 0
echo "==== LOO $HELD $ARCH FL=$FL done ===="
