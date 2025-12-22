#!/usr/bin/env bash
set -euo pipefail

dataset_name=APA_200MHz
version=rev4
seed=0
PA_backbone=transformer_encoder
PA_hidden_size=128
PA_num_layers=5
DPD_backbone=transformer_encoder
DPD_hidden_size=128
DPD_num_layers=5
frame_length=200
frame_stride=1
batch_size=64
batch_size_eval=256
n_epochs=50

# Train PA
printf '\033[32m==== Train PA model (seed=%s, backbone=%s) ====\033[0m\n' "${seed}" "${PA_backbone}"
python main.py \
  --dataset_name "${dataset_name}" \
  --step train_pa \
  --accelerator cuda \
  --devices 1 \
  --seed "${seed}" \
  --PA_backbone "${PA_backbone}" \
  --PA_hidden_size "${PA_hidden_size}" \
  --PA_num_layers "${PA_num_layers}" \
  --frame_length "${frame_length}" \
  --frame_stride "${frame_stride}" \
  --batch_size "${batch_size}" \
  --batch_size_eval "${batch_size_eval}" \
  --n_epochs "${n_epochs}" \
  --version "${version}" \
  2>&1 | tee terminal_log/${version}/pa.log

# Train DPD
printf '\033[32m==== Train DPD model (seed=%s, backbone=%s) ====\033[0m\n' "${seed}" "${DPD_backbone}"
python main.py \
  --dataset_name "${dataset_name}" \
  --step train_dpd \
  --accelerator cuda \
  --devices 0 \
  --seed "${seed}" \
  --PA_backbone "${PA_backbone}" \
  --PA_hidden_size "${PA_hidden_size}" \
  --PA_num_layers "${PA_num_layers}" \
  --DPD_backbone "${DPD_backbone}" \
  --DPD_hidden_size "${DPD_hidden_size}" \
  --DPD_num_layers "${DPD_num_layers}" \
  --frame_length "${frame_length}" \
  --frame_stride "${frame_stride}" \
  --batch_size "${batch_size}" \
  --batch_size_eval "${batch_size_eval}" \
  --n_epochs "${n_epochs}" \
  --version "${version}" \
  2>&1 | tee terminal_log/${version}/dpd.log

# Find the trained DPD model
pretrained_pattern="./save/${dataset_name}/train_dpd/PA_S_${seed}_M_${PA_backbone^^}_H_${PA_hidden_size}_F_${frame_length}_P_*/${version}/DPD_S_${seed}_M_${DPD_backbone^^}_H_${DPD_hidden_size}_F_${frame_length}_P_*.pt"
pretrained_model=""
if pretrained_model=$(ls -1t ${pretrained_pattern} 2>/dev/null | head -n1 || true); then
  printf '\033[32m==== Run DPD (model: %s) ====\033[0m\n' "${pretrained_model}"
  python main.py \
    --dataset_name "${dataset_name}" \
    --step run_dpd \
    --accelerator cuda \
    --devices 1 \
    --seed "${seed}" \
    --PA_backbone "${PA_backbone}" \
    --PA_hidden_size "${PA_hidden_size}" \
    --PA_num_layers "${PA_num_layers}" \
    --DPD_backbone "${DPD_backbone}" \
    --DPD_hidden_size "${DPD_hidden_size}" \
    --DPD_num_layers "${DPD_num_layers}" \
    --frame_length "${frame_length}" \
    --frame_stride "${frame_stride}" \
    --batch_size "${batch_size}" \
    --batch_size_eval "${batch_size_eval}" \
    --n_epochs "${n_epochs}" \
    --pretrained_model "${pretrained_model}" \
    --version "${version}" \
    2>&1 | tee terminal_log/${version}/run.log
else
  printf '\033[31m[ERROR] Pretrained DPD model not found for pattern %s. Skipping run_dpd.\033[0m\n' "${pretrained_pattern}" >&2
  exit 1
fi