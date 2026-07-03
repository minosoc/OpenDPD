#!/usr/bin/env bash
# Reproduce OpenDPD paper's DGRU training on DPA_200MHz exactly.
#
# Paper quote:
#   "38,400 samples of 200 MHz OFDM signals sampled at 800 MHz rate.
#    Training is done using AdamW for 100 epochs with ReduceLROnPlateau,
#    starting at 1e-3 with batch size 64, frame length 50, stride 1."
#
# Usage: bash bash_scripts/paper_dgru_dpa200.sh [gpu_id]

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

devices="${1:-0}"

dataset_name=DPA_200MHz
accelerator=cuda

# Paper config
frame_length=50
frame_stride=1
loss_type=l2
opt_type=adamw
batch_size=64
batch_size_eval=256
n_epochs=100
lr_schedule=1        # ReduceLROnPlateau enabled
lr=1e-3
lr_end=1e-6
decay_factor=0.5
patience=10

# DGRU PA / DPD hidden sizes — paper Table II iso-parameter (486 params each)
PA_backbone=dgru
PA_hidden_size=8
PA_num_layers=1
DPD_backbone=dgru
DPD_hidden_size=8
DPD_num_layers=1

seed=0

common_args=(
  --dataset_name "${dataset_name}"
  --accelerator "${accelerator}"
  --devices "${devices}"
  --frame_length "${frame_length}"
  --frame_stride "${frame_stride}"
  --loss_type "${loss_type}"
  --opt_type "${opt_type}"
  --batch_size "${batch_size}"
  --batch_size_eval "${batch_size_eval}"
  --n_epochs "${n_epochs}"
  --lr_schedule "${lr_schedule}"
  --lr "${lr}"
  --lr_end "${lr_end}"
  --decay_factor "${decay_factor}"
  --patience "${patience}"
  --seed "${seed}"
  --PA_backbone "${PA_backbone}"
  --PA_hidden_size "${PA_hidden_size}"
  --PA_num_layers "${PA_num_layers}"
  --DPD_backbone "${DPD_backbone}"
  --DPD_hidden_size "${DPD_hidden_size}"
  --DPD_num_layers "${DPD_num_layers}"
)

printf '\033[32m==== [%s/DGRU paper] train_pa ====\033[0m\n' "${dataset_name}"
python main.py --step train_pa "${common_args[@]}"

printf '\033[32m==== [%s/DGRU paper] train_dpd ====\033[0m\n' "${dataset_name}"
python main.py --step train_dpd "${common_args[@]}"

printf '\033[32m==== [%s/DGRU paper] run_dpd ====\033[0m\n' "${dataset_name}"
python main.py --step run_dpd "${common_args[@]}"

printf '\033[32m==== [%s/DGRU paper] plot (compare without/with DPD) ====\033[0m\n' "${dataset_name}"
python main.py --step plot "${common_args[@]}" || echo "  (plot step optional — ignore failure)"
