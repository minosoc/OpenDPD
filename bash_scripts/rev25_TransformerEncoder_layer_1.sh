#!/usr/bin/env bash
set -euo pipefail

# 첫 번째 인자로 실행할 step을 받음 (기본값: all)
step="${1:-all}"

# naming
dataset_name=APA_200MHz
version=rev25

# backbone settings
PA_backbone=transformer_encoder
PA_hidden_size=23
PA_num_layers=1
DPD_backbone=transformer_encoder
DPD_hidden_size=15
DPD_num_layers=1
d_model=128
n_heads=4
d_ff=512
dropout_ff=0.1
dropout_attn=0.1
thx=0.01
thh=0.05

# training settings
devices=1
n_epochs=2000
lr=1e-3
lr_schedule=1
lr_end=1e-5
decay_factor=0.5
grad_clip_val=0.1
warmup_steps=1000
weight_decay=0.01

# dataset settings
seed=0
frame_length=200
frame_stride=1
batch_size=64
batch_size_eval=256

# validate step
valid_steps=("all" "train_pa" "run_pa" "train_dpd" "run_dpd")
if [[ ! " ${valid_steps[@]} " =~ " ${step} " ]]; then
    printf '\033[31mError: Invalid step "%s". Valid steps are: %s\033[0m\n' "${step}" "${valid_steps[*]}"
    exit 1
fi

# Train PA
if [[ "${step}" == "all" || "${step}" == "train_pa" ]]; then
    printf '\033[32m==== Train PA model (seed=%s, backbone=%s) ====\033[0m\n' "${seed}" "${PA_backbone}"
    python main.py \
      --dataset_name "${dataset_name}" \
      --version "${version}" \
      --step train_pa \
      --accelerator cuda \
      --devices "${devices}" \
      --frame_length "${frame_length}" \
      --frame_stride "${frame_stride}" \
      --seed "${seed}" \
      --batch_size "${batch_size}" \
      --batch_size_eval "${batch_size_eval}" \
      --n_epochs "${n_epochs}" \
      --lr_schedule "${lr_schedule}" \
      --lr "${lr}" \
      --lr_end "${lr_end}" \
      --decay_factor "${decay_factor}" \
      --grad_clip_val "${grad_clip_val}" \
      --warmup_steps "${warmup_steps}" \
      --weight_decay "${weight_decay}" \
      --PA_backbone "${PA_backbone}" \
      --PA_hidden_size "${PA_hidden_size}" \
      --PA_num_layers "${PA_num_layers}" \
      --d_model "${d_model}" \
      --n_heads "${n_heads}" \
      --d_ff "${d_ff}" \
      --dropout_ff "${dropout_ff}" \
      --dropout_attn "${dropout_attn}" \
      --thx "${thx}" \
      --thh "${thh}" \
      2>&1
fi

# Run PA
if [[ "${step}" == "all" || "${step}" == "run_pa" ]]; then
    printf '\033[32m==== Run PA (backbone: %s) ====\033[0m\n' "${PA_backbone}"
    python main.py \
      --dataset_name "${dataset_name}" \
      --version "${version}" \
      --step run_pa \
      --accelerator cuda \
      --devices "${devices}" \
      --frame_length "${frame_length}" \
      --frame_stride "${frame_stride}" \
      --seed "${seed}" \
      --batch_size "${batch_size}" \
      --batch_size_eval "${batch_size_eval}" \
      --n_epochs "${n_epochs}" \
      --lr_schedule "${lr_schedule}" \
      --lr "${lr}" \
      --lr_end "${lr_end}" \
      --decay_factor "${decay_factor}" \
      --grad_clip_val "${grad_clip_val}" \
      --warmup_steps "${warmup_steps}" \
      --weight_decay "${weight_decay}" \
      --PA_backbone "${PA_backbone}" \
      --PA_hidden_size "${PA_hidden_size}" \
      --PA_num_layers "${PA_num_layers}" \
      --DPD_backbone "${DPD_backbone}" \
      --DPD_hidden_size "${DPD_hidden_size}" \
      --DPD_num_layers "${DPD_num_layers}" \
      --d_model "${d_model}" \
      --n_heads "${n_heads}" \
      --d_ff "${d_ff}" \
      --dropout_ff "${dropout_ff}" \
      --dropout_attn "${dropout_attn}" \
      --thx "${thx}" \
      --thh "${thh}" \
      2>&1
fi

# Train DPD
if [[ "${step}" == "all" || "${step}" == "train_dpd" ]]; then
    printf '\033[32m==== Train DPD model (seed=%s, backbone=%s) ====\033[0m\n' "${seed}" "${DPD_backbone}"
    python main.py \
      --dataset_name "${dataset_name}" \
      --version "${version}" \
      --step train_dpd \
      --accelerator cuda \
      --devices "${devices}" \
      --frame_length "${frame_length}" \
      --frame_stride "${frame_stride}" \
      --seed "${seed}" \
      --batch_size "${batch_size}" \
      --batch_size_eval "${batch_size_eval}" \
      --n_epochs "${n_epochs}" \
      --lr_schedule "${lr_schedule}" \
      --lr "${lr}" \
      --lr_end "${lr_end}" \
      --decay_factor "${decay_factor}" \
      --grad_clip_val "${grad_clip_val}" \
      --warmup_steps "${warmup_steps}" \
      --weight_decay "${weight_decay}" \
      --PA_backbone "${PA_backbone}" \
      --PA_hidden_size "${PA_hidden_size}" \
      --PA_num_layers "${PA_num_layers}" \
      --DPD_backbone "${DPD_backbone}" \
      --DPD_hidden_size "${DPD_hidden_size}" \
      --DPD_num_layers "${DPD_num_layers}" \
      --d_model "${d_model}" \
      --n_heads "${n_heads}" \
      --d_ff "${d_ff}" \
      --dropout_ff "${dropout_ff}" \
      --dropout_attn "${dropout_attn}" \
      --thx "${thx}" \
      --thh "${thh}" \
      2>&1
fi

# Run DPD
if [[ "${step}" == "all" || "${step}" == "run_dpd" ]]; then
    printf '\033[32m==== Run DPD (backbone: %s) ====\033[0m\n' "${DPD_backbone}"
    python main.py \
      --dataset_name "${dataset_name}" \
      --version "${version}" \
      --step run_dpd \
      --accelerator cuda \
      --devices "${devices}" \
      --frame_length "${frame_length}" \
      --frame_stride "${frame_stride}" \
      --seed "${seed}" \
      --batch_size "${batch_size}" \
      --batch_size_eval "${batch_size_eval}" \
      --n_epochs "${n_epochs}" \
      --lr_schedule "${lr_schedule}" \
      --lr "${lr}" \
      --lr_end "${lr_end}" \
      --decay_factor "${decay_factor}" \
      --grad_clip_val "${grad_clip_val}" \
      --warmup_steps "${warmup_steps}" \
      --weight_decay "${weight_decay}" \
      --PA_backbone "${PA_backbone}" \
      --PA_hidden_size "${PA_hidden_size}" \
      --PA_num_layers "${PA_num_layers}" \
      --DPD_backbone "${DPD_backbone}" \
      --DPD_hidden_size "${DPD_hidden_size}" \
      --DPD_num_layers "${DPD_num_layers}" \
      --d_model "${d_model}" \
      --n_heads "${n_heads}" \
      --d_ff "${d_ff}" \
      --dropout_ff "${dropout_ff}" \
      --dropout_attn "${dropout_attn}" \
      --thx "${thx}" \
      --thh "${thh}" \
      2>&1
fi