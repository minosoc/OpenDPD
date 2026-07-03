#!/usr/bin/env bash
# Mamba diagnostic: FL in {50,200} x variants {lr3e-4, grad-clip1.0, seed1}.
# Distribute 6 runs across GPUs 0,2,7 (sequential per GPU).
set -uo pipefail
cd /home/mkiuyh/workspace/LLM-basedDPD/OpenDPD
source /home/mkiuyh/miniconda3/etc/profile.d/conda.sh && conda activate LLMDPD
SD=bash_scripts; mkdir -p logs

# variant: "lr clip seed"  (baseline was 1e-3 200 0)
VARIANTS=("3e-4 200 0" "1e-3 1.0 0" "1e-3 200 1")
GPUS=(0 2 7)
FLS=(50 200)

jobs=()
for fl in "${FLS[@]}"; do for v in "${VARIANTS[@]}"; do jobs+=("$fl $v"); done; done
declare -A Q; i=0
for j in "${jobs[@]}"; do g=${GPUS[$((i % ${#GPUS[@]}))]}; Q[$g]+="$j"$'\n'; i=$((i+1)); done

pids=()
for g in "${GPUS[@]}"; do
  (
    while IFS= read -r line; do
      [ -z "$line" ] && continue
      set -- $line; fl=$1; lr=$2; clip=$3; seed=$4
      tag="LR${lr}_CLIP${clip}_SEED${seed}"
      echo "[$(date +%H:%M:%S)][GPU $g] START FL=$fl $tag"
      bash "$SD/train_mamba_diag.sh" "$fl" "$lr" "$clip" "$seed" "$g" 4 \
        > "logs/mamba_diag_FL${fl}_${tag}.log" 2>&1 \
        && echo "[$(date +%H:%M:%S)][GPU $g] DONE FL=$fl $tag" \
        || echo "[$(date +%H:%M:%S)][GPU $g] FAIL FL=$fl $tag"
    done <<< "${Q[$g]}"
  ) &
  pids+=($!)
done
wait "${pids[@]}"
echo "[$(date +%H:%M:%S)] ==== mamba diag done ===="
