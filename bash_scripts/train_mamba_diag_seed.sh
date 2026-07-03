#!/usr/bin/env bash
# Seed-aware Mamba diag (seed in PA/DPD filename). lr=1e-3, clip=200.
# Usage: train_mamba_diag_seed.sh <FL> <seed> <gpu> [stride]
set -uo pipefail
cd /home/mkiuyh/workspace/LLM-basedDPD/OpenDPD
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh && conda activate LLMDPD
FL="${1:?}"; SEED="${2:?}"; gpu="${3:-7}"; STRIDE="${4:-4}"
if   [ "$FL" -le 100 ]; then BS=64
elif [ "$FL" -le 200 ]; then BS=64
elif [ "$FL" -le 500 ]; then BS=32
else BS=16; fi

# PA for this seed/FL -> point at the real (seed-0) universal PA
ln -sf PA_S_0_M_DGRU_H_8_F_50_P_486.pt "save/GaN_combined/train_pa/PA_S_${SEED}_M_DGRU_H_8_F_${FL}_P_486.pt"

TAG="LR1e-3_CLIP200_SEED${SEED}"
echo "==== mamba diag-seed FL=$FL seed=$SEED GPU=$gpu ===="
python main.py --step train_dpd --dataset_name GaN_combined \
  --accelerator cuda --devices "$gpu" \
  --frame_length "$FL" --frame_stride "$STRIDE" --batch_size "$BS" --batch_size_eval 256 \
  --n_epochs 100 --opt_type adamw --lr 1e-3 --lr_end 1e-6 --lr_schedule 1 --decay_factor 0.5 --patience 10 \
  --grad_clip_val 200 --seed "$SEED" \
  --PA_backbone dgru --PA_hidden_size 8 --PA_num_layers 1 \
  --DPD_backbone mamba --DPD_hidden_size 6 --DPD_num_layers 1 \
  --mamba_d_state 4 --mamba_d_conv 4 --mamba_expand 2 \
  --last_token_loss 1 --eval_match_train_len 0
# copy produced (seed-specific) ckpt into canonical seed-0 dir with tag for unified eval
SRC=$(ls save/GaN_combined/train_dpd/PA_S_${SEED}_M_DGRU_H_8_F_${FL}/DPD_S_${SEED}_M_MAMBA_H_6_F_${FL}_P_*_LTL_1.pt 2>/dev/null | grep -vE "_LR|_CLIP|_ORIG" | head -1)
DST="save/GaN_combined/train_dpd/PA_S_0_M_DGRU_H_8_F_${FL}/DPD_S_0_M_MAMBA_H_6_F_${FL}_P_548_LTL_1_${TAG}.pt"
[ -n "$SRC" ] && cp "$SRC" "$DST" && echo "tagged: $DST"
echo "==== diag-seed FL=$FL seed=$SEED done ===="
