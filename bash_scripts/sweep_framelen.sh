#!/usr/bin/env bash
# Frame-length sweep: DGRU vs Transformer (no PE / with PE)
#
# - PA fixed = DGRU H=8 (already trained, save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt)
# - DPD: backbone × frame_length sweep
# - Train and eval use SAME frame_length (--eval_match_train_len 1)
#
# Each (backbone, frame_length) is one main.py call.
#
# Usage: bash bash_scripts/sweep_framelen.sh <backbone_tag> <frame_length> <gpu_id>
#   backbone_tag ∈ {dgru, attn, attn_pe}

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

# Activate conda env (child bash shells don't inherit conda functions)
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh
conda activate LLMDPD

bb_tag="${1:?usage: $0 <backbone_tag> <frame_length> <gpu_id>}"
FL="${2:?usage: $0 <backbone_tag> <frame_length> <gpu_id>}"
devices="${3:-0}"

# Common — paper config
dataset_name=DPA_200MHz
n_epochs=100
lr=1e-3
lr_end=1e-6
lr_schedule=1
opt_type=adamw
seed=0
patience=10
decay_factor=0.5
batch_size_eval=256

# Batch size scales down with frame_length to avoid OOM on attention.
# baseline B=64 at FL=50. For attention O(T^2 * B) memory.
if   [ "$FL" -le 100 ];  then BS=64
elif [ "$FL" -le 500 ];  then BS=32
elif [ "$FL" -le 1000 ]; then BS=16
elif [ "$FL" -le 2000 ]; then BS=8
elif [ "$FL" -le 5000 ]; then BS=4
else                          BS=2
fi

# Need PA model trained at frame_length=50 to be reused regardless of new FL.
# train_dpd loads PA via gen_pa_model_id which uses frame_length in the ID.
# We work around: the PA at FL=50 is already trained; reuse by symlinking.
PA_SRC="save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt"
PA_DST="save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_${FL}_P_486.pt"
if [ ! -f "${PA_DST}" ] && [ "${FL}" != "50" ]; then
    ln -sf "$(basename ${PA_SRC})" "${PA_DST}"
fi

# Common args
common=(
  --dataset_name "${dataset_name}"
  --accelerator cuda
  --devices "${devices}"
  --frame_length "${FL}"
  --frame_stride 1
  --batch_size "${BS}"
  --batch_size_eval "${batch_size_eval}"
  --n_epochs "${n_epochs}"
  --opt_type "${opt_type}"
  --lr "${lr}"
  --lr_end "${lr_end}"
  --lr_schedule "${lr_schedule}"
  --decay_factor "${decay_factor}"
  --patience "${patience}"
  --seed "${seed}"
  --PA_backbone dgru
  --PA_hidden_size 8
  --PA_num_layers 1
  --eval_match_train_len 1
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
  *) echo "Unknown backbone_tag: ${bb_tag}"; exit 1 ;;
esac

printf '\033[32m==== [%s/%s/FL=%d/BS=%d] train_dpd ====\033[0m\n' "${dataset_name}" "${bb_tag}" "${FL}" "${BS}"
python main.py --step train_dpd "${common[@]}" "${dpd[@]}"
