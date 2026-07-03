#!/usr/bin/env bash
# Universal PA (DGRU H=8) on pooled GaN_combined.
set -euo pipefail
cd /home/mkiuyh/workspace/LLM-basedDPD/OpenDPD
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh && conda activate LLMDPD
devices="${1:-0}"
echo "==== GaN_combined universal PA (DGRU H=8) GPU=${devices} ===="
python main.py --step train_pa \
  --dataset_name GaN_combined \
  --accelerator cuda --devices "${devices}" \
  --frame_length 50 --frame_stride 1 \
  --batch_size 64 --batch_size_eval 256 \
  --n_epochs 100 --opt_type adamw \
  --lr 1e-3 --lr_end 1e-6 --lr_schedule 1 --decay_factor 0.5 --patience 10 \
  --seed 0 \
  --PA_backbone dgru --PA_hidden_size 8 --PA_num_layers 1 \
  --DPD_backbone dgru --DPD_hidden_size 8 --DPD_num_layers 1
echo "==== combined PA done ===="
ls -la save/GaN_combined/train_pa/ 2>/dev/null
