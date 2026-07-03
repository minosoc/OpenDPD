#!/usr/bin/env bash
# rev10e: GMP-stem k ∈ {21, 41} × lr sweep @ FL=200, LTL, no PE, d_ff=10.
# Usage: bash bash_scripts/train_rev10e.sh <K> <LR> <gpu_id>
# Filename gets LR{lr} suffix to distinguish.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh
conda activate LLMDPD

K="${1:?usage: $0 <K> <LR> <gpu_id>}"
LR="${2:?}"
devices="${3:-0}"
FL=200

# Make a tag-friendly LR string for log filename (e.g. 1e-3 → 1e-3)
LRTAG="${LR}"
LOGDIR="${REPO_ROOT}/logs"
mkdir -p "${LOGDIR}"

PA_SRC="save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt"
PA_DST="save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_${FL}_P_486.pt"
[ ! -f "${PA_DST}" ] && ln -sf "$(basename ${PA_SRC})" "${PA_DST}"

# Move/rename produced ckpt afterwards so different LRs don't overwrite each other
SRC_DIR="save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_${FL}"

echo "==== rev10e GMP-stem k=${K} lr=${LR} GPU=${devices} ===="
python main.py --step train_dpd \
  --dataset_name DPA_200MHz \
  --accelerator cuda --devices "${devices}" \
  --frame_length "${FL}" --frame_stride 1 \
  --batch_size 64 --batch_size_eval 256 \
  --n_epochs 100 --opt_type adamw \
  --lr "${LR}" --lr_end 1e-6 --lr_schedule 1 --decay_factor 0.5 --patience 10 \
  --seed 0 \
  --PA_backbone dgru --PA_hidden_size 8 --PA_num_layers 1 \
  --DPD_backbone transformer --DPD_hidden_size 6 --DPD_num_layers 1 \
  --n_heads 2 --d_ff 10 --use_pos_encoding 0 \
  --use_gmp_stem 1 --gmp_stem_kernel "${K}" \
  --last_token_loss 1 \
  --eval_match_train_len 0

# Tag the produced best ckpt with lr so different LRs do not overwrite
SRC_CKPT=$(ls "${SRC_DIR}"/DPD_S_0_M_TRANSFORMER_H_6_F_${FL}_*_LTL_1_PE_0_GMP_1_GSK_${K}.pt 2>/dev/null | head -1)
if [ -n "${SRC_CKPT}" ] && [ -f "${SRC_CKPT}" ]; then
    BASE=$(basename "${SRC_CKPT}" .pt)
    DST_CKPT="${SRC_DIR}/${BASE}_LR_${LRTAG}.pt"
    cp "${SRC_CKPT}" "${DST_CKPT}"
    echo "Saved tagged ckpt: ${DST_CKPT}"
fi
echo "==== rev10e k=${K} lr=${LR} done ===="
