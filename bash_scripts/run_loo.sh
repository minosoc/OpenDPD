#!/usr/bin/env bash
set -uo pipefail
cd /home/mkiuyh/workspace/LLM-basedDPD/OpenDPD
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh && conda activate LLMDPD
SD=bash_scripts; mkdir -p logs
GPUS=(0 3 4 7)
# (held arch FL)
jobs=()
for h in c24 c27 b20 b24; do
  jobs+=("$h dgru 200"); jobs+=("$h dgru 1000"); jobs+=("$h mamba 1000")
done
declare -A Q; i=0
for j in "${jobs[@]}"; do g=${GPUS[$((i%${#GPUS[@]}))]}; Q[$g]+="$j"$'\n'; i=$((i+1)); done
pids=()
for g in "${GPUS[@]}"; do
  ( while IFS= read -r line; do
      [ -z "$line" ] && continue; set -- $line; h=$1; a=$2; fl=$3
      echo "[$(date +%H:%M:%S)][GPU $g] START $h $a FL=$fl"
      bash "$SD/train_loo_dpd.sh" "$h" "$a" "$fl" "$g" 4 > "logs/loo_${h}_${a}_FL${fl}.log" 2>&1 \
        && echo "[$(date +%H:%M:%S)][GPU $g] DONE $h $a FL=$fl" || echo "[$(date +%H:%M:%S)][GPU $g] FAIL $h $a FL=$fl"
    done <<< "${Q[$g]}" ) & pids+=($!)
done
wait "${pids[@]}"; echo "[$(date +%H:%M:%S)] ==== LOO done ===="
