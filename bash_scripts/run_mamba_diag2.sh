#!/usr/bin/env bash
# Clean re-run of lr3e-4 / clip1.0 variants WITHOUT same-FL concurrency
# (canonical ckpt path has no lr/clip tag, so concurrent same-FL runs race).
# GPU0 = both FL50 variants serial; GPU2 = both FL200 variants serial.
set -uo pipefail
cd /home/mkiuyh/workspace/LLM-basedDPD/OpenDPD
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh && conda activate LLMDPD
SD=bash_scripts; mkdir -p logs

# remove contaminated tags
for fl in 50 200; do for t in LR3e-4_CLIP200_SEED0 LR1e-3_CLIP1.0_SEED0; do
  rm -f "save/GaN_combined/train_dpd/PA_S_0_M_DGRU_H_8_F_${fl}/DPD_S_0_M_MAMBA_H_6_F_${fl}_P_548_LTL_1_${t}.pt"
done; done

( echo "[GPU0] FL50 lr3e-4"; bash $SD/train_mamba_diag.sh 50 3e-4 200 0 0 4 > logs/mamba_diag2_FL50_LR3e-4.log 2>&1
  echo "[GPU0] FL50 clip1.0"; bash $SD/train_mamba_diag.sh 50 1e-3 1.0 0 0 4 > logs/mamba_diag2_FL50_CLIP1.0.log 2>&1
  echo "[GPU0] FL50 done" ) & p0=$!
( echo "[GPU2] FL200 lr3e-4"; bash $SD/train_mamba_diag.sh 200 3e-4 200 0 2 4 > logs/mamba_diag2_FL200_LR3e-4.log 2>&1
  echo "[GPU2] FL200 clip1.0"; bash $SD/train_mamba_diag.sh 200 1e-3 1.0 0 2 4 > logs/mamba_diag2_FL200_CLIP1.0.log 2>&1
  echo "[GPU2] FL200 done" ) & p2=$!
wait $p0 $p2
echo "==== diag2 (clean lr/clip) done ===="
