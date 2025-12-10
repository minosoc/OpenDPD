#!/usr/bin/env bash
# Run quantization model from epoch 2
# Usage: bash run_quant_epoch2.sh

cd /home/user/minho/OpenDPD

python main.py \
    --step run_dpd \
    --dataset_name DPA_200MHz \
    --version rev_qgru50_quant \
    --pretrained_model ./save/DPA_200MHz/train_dpd/PA_S_0_M_GRU_H_23_F_200/rev_qgru50_quant/quant_w8a8/DPD_S_0_M_QGRU_H_15_F_200_P_990.pt \
    --quant \
    --n_bits_w 8 \
    --n_bits_a 8 \
    --PA_backbone gru \
    --PA_hidden_size 23 \
    --PA_num_layers 1 \
    --DPD_backbone qgru \
    --DPD_hidden_size 15 \
    --DPD_num_layers 1 \
    --frame_length 200 \
    --seed 0 \
    --accelerator cuda \
    --devices 0

