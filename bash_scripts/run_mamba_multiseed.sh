#!/usr/bin/env bash
# Mamba multi-seed variance study: FL{50,200,1000} x seed{1,2,3,4}, stride 4.
# seed is in the save path (PA_S_<seed>/DPD_S_<seed>) -> no canonical-path race.
set -uo pipefail
cd /home/mkiuyh/workspace/LLM-basedDPD/OpenDPD
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh && conda activate LLMDPD
SD=bash_scripts; mkdir -p logs
GPUS=(0 3 4 7); FLS=(50 200 1000); SEEDS=(1 2 3 4)
jobs=(); for fl in "${FLS[@]}"; do for sd in "${SEEDS[@]}"; do jobs+=("$fl $sd"); done; done
declare -A Q; i=0
for j in "${jobs[@]}"; do g=${GPUS[$((i%${#GPUS[@]}))]}; Q[$g]+="$j"$'\n'; i=$((i+1)); done
pids=()
for g in "${GPUS[@]}"; do
  ( while IFS= read -r line; do
      [ -z "$line" ] && continue; set -- $line; fl=$1; sd=$2
      echo "[$(date +%H:%M:%S)][GPU $g] START FL=$fl seed=$sd"
      bash "$SD/train_mamba_diag_seed.sh" "$fl" "$sd" "$g" 4 > "logs/mamba_ms_FL${fl}_s${sd}.log" 2>&1 \
        && echo "[$(date +%H:%M:%S)][GPU $g] DONE FL=$fl seed=$sd" || echo "[$(date +%H:%M:%S)][GPU $g] FAIL FL=$fl seed=$sd"
    done <<< "${Q[$g]}" ) & pids+=($!)
done
wait "${pids[@]}"; echo "[$(date +%H:%M:%S)] ==== mamba multiseed done ===="
