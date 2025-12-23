#!/usr/bin/env bash
set -euo pipefail

dataset_name=APA_200MHz
version=rev1
PA_backbone=transformer_encoder
PA_num_layers=5
DPD_backbone=transformer_encoder
DPD_num_layers=5
d_model=1024
n_heads=8
d_ff=4096
dropout_ff=0.1
dropout_attn=0.1
n_epochs=10

seed=0
frame_length=200
frame_stride=1
batch_size=64
batch_size_eval=256

# Train PA
printf '\033[32m==== Train PA model (seed=%s, backbone=%s) ====\033[0m\n' "${seed}" "${PA_backbone}"
python main.py \
  --dataset_name "${dataset_name}" \
  --step train_pa \
  --accelerator cuda \
  --devices 1 \
  --frame_length "${frame_length}" \
  --frame_stride "${frame_stride}" \
  --seed "${seed}" \
  --batch_size "${batch_size}" \
  --batch_size_eval "${batch_size_eval}" \
  --n_epochs "${n_epochs}" \
  --PA_backbone "${PA_backbone}" \
  --PA_num_layers "${PA_num_layers}" \
  --d_model "${d_model}" \
  --n_heads "${n_heads}" \
  --d_ff "${d_ff}" \
  --dropout_ff "${dropout_ff}" \
  --dropout_attn "${dropout_attn}" \
  --version "${version}" 2>&1

# Train DPD
printf '\033[32m==== Train DPD model (seed=%s, backbone=%s) ====\033[0m\n' "${seed}" "${DPD_backbone}"
python main.py \
  --dataset_name "${dataset_name}" \
  --step train_dpd \
  --accelerator cuda \
  --devices 1 \
  --frame_length "${frame_length}" \
  --frame_stride "${frame_stride}" \
  --seed "${seed}" \
  --batch_size "${batch_size}" \
  --batch_size_eval "${batch_size_eval}" \
  --n_epochs "${n_epochs}" \
  --PA_backbone "${PA_backbone}" \
  --PA_num_layers "${PA_num_layers}" \
  --DPD_backbone "${DPD_backbone}" \
  --DPD_num_layers "${DPD_num_layers}" \
  --d_model "${d_model}" \
  --n_heads "${n_heads}" \
  --d_ff "${d_ff}" \
  --dropout_ff "${dropout_ff}" \
  --dropout_attn "${dropout_attn}" \
  --version "${version}" 2>&1

# Find the trained DPD model
printf '\033[32m==== Run DPD (backbone: %s) ====\033[0m\n' "${DPD_backbone}"
python main.py \
  --dataset_name "${dataset_name}" \
  --step run_dpd \
  --accelerator cuda \
  --devices 1 \
  --frame_length "${frame_length}" \
  --frame_stride "${frame_stride}" \
  --seed "${seed}" \
  --batch_size "${batch_size}" \
  --batch_size_eval "${batch_size_eval}" \
  --n_epochs "${n_epochs}" \
  --PA_backbone "${PA_backbone}" \
  --PA_num_layers "${PA_num_layers}" \
  --DPD_backbone "${DPD_backbone}" \
  --DPD_num_layers "${DPD_num_layers}" \
  --d_model "${d_model}" \
  --n_heads "${n_heads}" \
  --d_ff "${d_ff}" \
  --dropout_ff "${dropout_ff}" \
  --dropout_attn "${dropout_attn}" \
  --version "${version}" 2>&1