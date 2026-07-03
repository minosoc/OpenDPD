#!/usr/bin/env bash
# Mamba instability diagnostic on GaN_combined.
# Usage: train_mamba_diag.sh <FL> <lr> <grad_clip> <seed> <gpu> [stride]
set -uo pipefail
cd /home/mkiuyh/workspace/LLM-basedDPD/OpenDPD
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh && conda activate LLMDPD
FL="${1:?}"; LR="${2:?}"; CLIP="${3:?}"; SEED="${4:?}"; gpu="${5:-0}"; STRIDE="${6:-4}"

if   [ "$FL" -le 100 ]; then BS=64
elif [ "$FL" -le 200 ]; then BS=64
elif [ "$FL" -le 500 ]; then BS=32
else BS=16; fi

PA_SRC="save/GaN_combined/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt"
PA_DST="save/GaN_combined/train_pa/PA_S_0_M_DGRU_H_8_F_${FL}_P_486.pt"
[ "$FL" != "50" ] && [ ! -e "$PA_DST" ] && ln -sf "$(basename "$PA_SRC")" "$PA_DST"

TAG="LR${LR}_CLIP${CLIP}_SEED${SEED}"
echo "==== mamba diag FL=$FL ${TAG} stride=$STRIDE GPU=$gpu BS=$BS ===="
python main.py --step train_dpd --dataset_name GaN_combined \
  --accelerator cuda --devices "$gpu" \
  --frame_length "$FL" --frame_stride "$STRIDE" --batch_size "$BS" --batch_size_eval 256 \
  --n_epochs 100 --opt_type adamw \
  --lr "$LR" --lr_end 1e-6 --lr_schedule 1 --decay_factor 0.5 --patience 10 \
  --grad_clip_val "$CLIP" --seed "$SEED" \
  --PA_backbone dgru --PA_hidden_size 8 --PA_num_layers 1 \
  --DPD_backbone mamba --DPD_hidden_size 6 --DPD_num_layers 1 \
  --mamba_d_state 4 --mamba_d_conv 4 --mamba_expand 2 \
  --last_token_loss 1 --eval_match_train_len 0

SRC=$(ls save/GaN_combined/train_dpd/PA_S_0_M_DGRU_H_8_F_${FL}/DPD_S_0_M_MAMBA_H_6_F_${FL}_P_*_LTL_1.pt 2>/dev/null | grep -vE "_LR|_CLIP|_ORIG" | head -1)
if [ -n "$SRC" ]; then cp "$SRC" "${SRC%.pt}_${TAG}.pt"; echo "tagged: ${SRC%.pt}_${TAG}.pt"; fi
echo "==== mamba diag FL=$FL ${TAG} done ===="
