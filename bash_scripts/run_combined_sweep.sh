#!/usr/bin/env bash
# Master: wait for universal PA, then run DGRU/Transformer/Mamba x FL{50,100,200,500,1000}
# LTL DPD on GaN_combined, distributed across GPUs (3 runs each, sequential per GPU).
set -uo pipefail
cd /home/mkiuyh/workspace/LLM-basedDPD/OpenDPD
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh && conda activate LLMDPD
SD=bash_scripts
mkdir -p logs

PA_CKPT="save/GaN_combined/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt"
echo "[$(date +%H:%M:%S)] waiting for universal PA ..."
while ! grep -q "combined PA done" logs/combined_pa.log 2>/dev/null; do sleep 20; done
[ -e "$PA_CKPT" ] || { echo "PA ckpt missing! abort"; exit 1; }
echo "[$(date +%H:%M:%S)] PA ready -> launching DPD sweep"

GPUS=(0 2 3 4 7)
ARCHS=(dgru transformer mamba)
FLS=(50 100 200 500 1000)
STRIDE=4; EP=100

jobs=(); for a in "${ARCHS[@]}"; do for fl in "${FLS[@]}"; do jobs+=("$a $fl"); done; done
declare -A Q; i=0
for j in "${jobs[@]}"; do g=${GPUS[$((i % ${#GPUS[@]}))]}; Q[$g]+="$j"$'\n'; i=$((i+1)); done

pids=()
for g in "${GPUS[@]}"; do
  (
    while IFS= read -r line; do
      [ -z "$line" ] && continue
      set -- $line; a=$1; fl=$2
      echo "[$(date +%H:%M:%S)][GPU $g] START $a FL=$fl"
      bash "$SD/train_combined_dpd.sh" "$a" "$fl" "$g" "$STRIDE" "$EP" \
        > "logs/combined_${a}_FL${fl}.log" 2>&1 \
        && echo "[$(date +%H:%M:%S)][GPU $g] DONE $a FL=$fl" \
        || echo "[$(date +%H:%M:%S)][GPU $g] FAIL $a FL=$fl"
    done <<< "${Q[$g]}"
  ) &
  pids+=($!)
done
wait "${pids[@]}"
echo "[$(date +%H:%M:%S)] ==== ALL combined DPD runs done ===="
