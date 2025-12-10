#!/usr/bin/env bash
# Script to run PA training, DPD training, and validation with version management
# Usage: ./run_version.sh <version_name> <version_description>
# Example: ./run_version.sh rev6 "target_gain_all_data"

set -euo pipefail

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <version_name> <version_description>"
    echo "Example: $0 rev6 'target_gain_all_data'"
    exit 1
fi

VERSION_NAME="$1"
VERSION_DESC="$2"
LOG_DIR="terminal_log/${VERSION_NAME}_${VERSION_DESC}"

# Create log directory
mkdir -p "${LOG_DIR}"

# Get script directory and repo root
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

# Python binary
PYTHON_BIN=${PYTHON:-python}

# Default arguments (can be overridden by environment variables)
DATASET_NAME=${DATASET_NAME:-DPA_200MHz}
ACCELERATOR=${ACCELERATOR:-cuda}
DEVICES=${DEVICES:-0}
SEED=${SEED:-0}
PA_BACKBONE=${PA_BACKBONE:-gru}
PA_HIDDEN_SIZE=${PA_HIDDEN_SIZE:-23}
PA_NUM_LAYERS=${PA_NUM_LAYERS:-1}
DPD_BACKBONE=${DPD_BACKBONE:-gru}
DPD_HIDDEN_SIZE=${DPD_HIDDEN_SIZE:-15}
DPD_NUM_LAYERS=${DPD_NUM_LAYERS:-1}
FRAME_LENGTH=${FRAME_LENGTH:-200}
N_EPOCHS=${N_EPOCHS:-50}
# Quantization settings
QUANT=${QUANT:-false}
QUANT_BITS_W=${QUANT_BITS_W:-8}
QUANT_BITS_A=${QUANT_BITS_A:-8}
QUANT_DIR_LABEL=${QUANT_DIR_LABEL:-}

echo "=========================================="
echo "OpenDPD Version Run Script"
echo "=========================================="
echo "Version: ${VERSION_NAME}"
echo "Description: ${VERSION_DESC}"
echo "Log Directory: ${LOG_DIR}"
if [ "${QUANT}" = "true" ]; then
    echo "Quantization: Enabled (W:${QUANT_BITS_W} bits, A:${QUANT_BITS_A} bits)"
    echo "Epochs: ${N_EPOCHS}"
fi
echo "=========================================="
echo ""

####################################################################################################
# Helper function to find PA model
####################################################################################################
find_pa_model() {
    local pa_model_dir="./save/${DATASET_NAME}/train_pa"
    if [ -n "${VERSION_NAME}" ]; then
        pa_model_dir="${pa_model_dir}/${VERSION_NAME}"
    fi
    
    # Check if directory exists
    if [ ! -d "${pa_model_dir}" ]; then
        return 1
    fi
    
    # Find the most recent .pt file in the PA model directory
    local pa_model_file=""
    if command -v find >/dev/null 2>&1 && find --version >/dev/null 2>&1; then
        # Try find with -printf (GNU find)
        pa_model_file=$(find "${pa_model_dir}" -name "PA_*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n1 | cut -d' ' -f2- 2>/dev/null || echo "")
    fi
    
    # Alternative method if find -printf is not available or failed
    if [ -z "${pa_model_file}" ]; then
        pa_model_file=$(ls -1t "${pa_model_dir}"/PA_*.pt 2>/dev/null | head -n1 || echo "")
    fi
    
    if [ -n "${pa_model_file}" ] && [ -f "${pa_model_file}" ]; then
        echo "${pa_model_file}"
        return 0
    else
        return 1
    fi
}

####################################################################################################
# 1. PA Modeling
####################################################################################################
echo "####################################################################################################"
echo "# Step: Check PA Model                                                                              #"
echo "####################################################################################################"

# Check if PA model already exists
PA_MODEL_FILE=$(find_pa_model 2>/dev/null || echo "")
if [ -n "${PA_MODEL_FILE}" ] && [ -f "${PA_MODEL_FILE}" ]; then
    echo "Found existing PA model: ${PA_MODEL_FILE}"
    echo "Skipping PA training..."
else
    echo "PA model not found. Starting PA training..."
    echo ""
    echo "####################################################################################################"
    echo "# Step: Train PA                                                                                   #"
    echo "####################################################################################################"
    "${PYTHON_BIN}" main.py \
        --dataset_name "${DATASET_NAME}" \
        --step train_pa \
        --version "${VERSION_NAME}" \
        --accelerator "${ACCELERATOR}" \
        --devices "${DEVICES}" \
        --seed "${SEED}" \
        --PA_backbone "${PA_BACKBONE}" \
        --PA_hidden_size "${PA_HIDDEN_SIZE}" \
        --PA_num_layers "${PA_NUM_LAYERS}" \
        --frame_length "${FRAME_LENGTH}" \
        2>&1 | tee "${LOG_DIR}/pa.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: PA training failed!"
        exit 1
    fi

    # Wait a moment for file system to sync
    sleep 1

    # Find the PA model file after training
    PA_MODEL_FILE=$(find_pa_model 2>/dev/null || echo "")
    if [ -z "${PA_MODEL_FILE}" ] || [ ! -f "${PA_MODEL_FILE}" ]; then
        echo "ERROR: PA model not found after training in ./save/${DATASET_NAME}/train_pa/${VERSION_NAME}"
        echo "Please check if PA training completed successfully."
        exit 1
    fi
    echo "Found PA model: ${PA_MODEL_FILE}"
fi

####################################################################################################
# Helper function to find DPD model
####################################################################################################
find_dpd_model() {
    local pa_model_file="$1"
    # Extract PA model ID from PA model file path (without .pt extension and PA_ prefix)
    local pa_model_basename=$(basename "${pa_model_file}" .pt)
    # Remove PA_ prefix and _P_* suffix to get the model ID part without parameter count
    # (e.g., PA_S_0_M_GRU_H_23_F_200_P_1911 -> PA_S_0_M_GRU_H_23_F_200)
    # This matches how gen_pa_model_id() generates the directory path (without P parameter)
    local pa_model_dir_name=$(echo "${pa_model_basename}" | sed 's/_P_[0-9]*$//')
    
    # DPD models are stored under PA model directory (without P parameter in directory name)
    local dpd_model_dir="./save/${DATASET_NAME}/train_dpd/${pa_model_dir_name}"
    if [ -n "${VERSION_NAME}" ]; then
        dpd_model_dir="${dpd_model_dir}/${VERSION_NAME}"
    fi
    
    # Check if directory exists
    if [ ! -d "${dpd_model_dir}" ]; then
        return 1
    fi
    
    # Find the most recent .pt file in the DPD model directory
    local dpd_model_file=""
    if command -v find >/dev/null 2>&1 && find --version >/dev/null 2>&1; then
        # Try find with -printf (GNU find)
        dpd_model_file=$(find "${dpd_model_dir}" -name "DPD_*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n1 | cut -d' ' -f2- 2>/dev/null || echo "")
    fi
    
    # Alternative method if find -printf is not available or failed
    if [ -z "${dpd_model_file}" ]; then
        dpd_model_file=$(ls -1t "${dpd_model_dir}"/DPD_*.pt 2>/dev/null | head -n1 || echo "")
    fi
    
    if [ -n "${dpd_model_file}" ] && [ -f "${dpd_model_file}" ]; then
        echo "${dpd_model_file}"
        return 0
    else
        return 1
    fi
}

####################################################################################################
# 2. DPD Learning
####################################################################################################
echo ""
echo "####################################################################################################"
echo "# Step: Check DPD Model                                                                             #"
echo "####################################################################################################"

# Check if DPD model already exists
DPD_MODEL_FILE=$(find_dpd_model "${PA_MODEL_FILE}" 2>/dev/null || echo "")
if [ -n "${DPD_MODEL_FILE}" ] && [ -f "${DPD_MODEL_FILE}" ]; then
    echo "Found existing DPD model: ${DPD_MODEL_FILE}"
    echo "Skipping DPD training..."
else
    echo "DPD model not found. Starting DPD training..."
    echo ""
    echo "####################################################################################################"
    echo "# Step: Train DPD                                                                                  #"
    echo "####################################################################################################"
    "${PYTHON_BIN}" main.py \
        --dataset_name "${DATASET_NAME}" \
        --step train_dpd \
        --version "${VERSION_NAME}" \
        --accelerator "${ACCELERATOR}" \
        --devices "${DEVICES}" \
        --seed "${SEED}" \
        --PA_backbone "${PA_BACKBONE}" \
        --PA_hidden_size "${PA_HIDDEN_SIZE}" \
        --PA_num_layers "${PA_NUM_LAYERS}" \
        --DPD_backbone "${DPD_BACKBONE}" \
        --DPD_hidden_size "${DPD_HIDDEN_SIZE}" \
        --DPD_num_layers "${DPD_NUM_LAYERS}" \
        --frame_length "${FRAME_LENGTH}" \
        --n_epochs "${N_EPOCHS}" \
        2>&1 | tee "${LOG_DIR}/dpd.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: DPD training failed!"
        exit 1
    fi

    # Wait a moment for file system to sync
    sleep 1

    # Find the DPD model file after training
    DPD_MODEL_FILE=$(find_dpd_model "${PA_MODEL_FILE}" 2>/dev/null || echo "")
    if [ -z "${DPD_MODEL_FILE}" ] || [ ! -f "${DPD_MODEL_FILE}" ]; then
        echo "ERROR: DPD model not found after training"
        echo "Please check if DPD training completed successfully."
        exit 1
    fi
    echo "Found DPD model: ${DPD_MODEL_FILE}"
fi

####################################################################################################
# Helper function to find quantized DPD model
####################################################################################################
find_quant_dpd_model() {
    local pa_model_file="$1"
    local quant_dir_label_to_use="$2"
    local pa_model_basename=$(basename "${pa_model_file}" .pt)
    local pa_model_dir_name=$(echo "${pa_model_basename}" | sed 's/_P_[0-9]*$//')
    
    local dpd_model_dir="./save/${DATASET_NAME}/train_dpd/${pa_model_dir_name}"
    if [ -n "${VERSION_NAME}" ]; then
        dpd_model_dir="${dpd_model_dir}/${VERSION_NAME}"
    fi
    if [ -n "${quant_dir_label_to_use}" ]; then
        dpd_model_dir="${dpd_model_dir}/${quant_dir_label_to_use}"
    fi
    
    # Check if directory exists
    if [ ! -d "${dpd_model_dir}" ]; then
        return 1
    fi
    
    # Find the most recent .pt file in the quantized DPD model directory
    local dpd_model_file=""
    if command -v find >/dev/null 2>&1 && find --version >/dev/null 2>&1; then
        dpd_model_file=$(find "${dpd_model_dir}" -name "DPD_*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n1 | cut -d' ' -f2- 2>/dev/null || echo "")
    fi
    
    if [ -z "${dpd_model_file}" ]; then
        dpd_model_file=$(ls -1t "${dpd_model_dir}"/DPD_*.pt 2>/dev/null | head -n1 || echo "")
    fi
    
    if [ -n "${dpd_model_file}" ] && [ -f "${dpd_model_file}" ]; then
        echo "${dpd_model_file}"
        return 0
    else
        return 1
    fi
}

####################################################################################################
# 2.5. Quantization DPD Learning (if enabled)
####################################################################################################
if [ "${QUANT}" = "true" ]; then
    echo ""
    echo "####################################################################################################"
    echo "# Step: Check Quantized DPD Model                                                                  #"
    echo "####################################################################################################"
    
    # Build quant_dir_label if not provided
    QUANT_DIR_LABEL_TO_USE="${QUANT_DIR_LABEL}"
    if [ -z "${QUANT_DIR_LABEL_TO_USE}" ]; then
        QUANT_DIR_LABEL_TO_USE="quant_w${QUANT_BITS_W}a${QUANT_BITS_A}"
    fi
    
    # Check if quantized DPD model already exists
    QUANT_DPD_MODEL_FILE=$(find_quant_dpd_model "${PA_MODEL_FILE}" "${QUANT_DIR_LABEL_TO_USE}" 2>/dev/null || echo "")
    if [ -n "${QUANT_DPD_MODEL_FILE}" ] && [ -f "${QUANT_DPD_MODEL_FILE}" ]; then
        echo "Found existing quantized DPD model: ${QUANT_DPD_MODEL_FILE}"
        echo "Skipping quantized DPD training..."
    else
        echo "Quantized DPD model not found. Starting quantized DPD training..."
        echo ""
        echo "####################################################################################################"
        echo "# Step: Train Quantized DPD                                                                       #"
        echo "####################################################################################################"
        
        "${PYTHON_BIN}" main.py \
            --dataset_name "${DATASET_NAME}" \
            --step train_dpd \
            --version "${VERSION_NAME}" \
            --accelerator "${ACCELERATOR}" \
            --devices "${DEVICES}" \
            --seed "${SEED}" \
            --PA_backbone "${PA_BACKBONE}" \
            --PA_hidden_size "${PA_HIDDEN_SIZE}" \
            --PA_num_layers "${PA_NUM_LAYERS}" \
            --DPD_backbone "${DPD_BACKBONE}" \
            --DPD_hidden_size "${DPD_HIDDEN_SIZE}" \
            --DPD_num_layers "${DPD_NUM_LAYERS}" \
            --frame_length "${FRAME_LENGTH}" \
            --n_epochs "${N_EPOCHS}" \
            --quant \
            --n_bits_w "${QUANT_BITS_W}" \
            --n_bits_a "${QUANT_BITS_A}" \
            --pretrained_model "${DPD_MODEL_FILE}" \
            --quant_dir_label "${QUANT_DIR_LABEL_TO_USE}" \
            2>&1 | tee "${LOG_DIR}/dpd_quant.log"
        
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "ERROR: Quantized DPD training failed!"
            exit 1
        fi
        
        # Wait a moment for file system to sync
        sleep 1
        
        # Find the quantized DPD model file after training
        QUANT_DPD_MODEL_FILE=$(find_quant_dpd_model "${PA_MODEL_FILE}" "${QUANT_DIR_LABEL_TO_USE}" 2>/dev/null || echo "")
        if [ -z "${QUANT_DPD_MODEL_FILE}" ] || [ ! -f "${QUANT_DPD_MODEL_FILE}" ]; then
            echo "ERROR: Quantized DPD model not found after training"
            echo "Please check if quantized DPD training completed successfully."
            exit 1
        fi
        echo "Found quantized DPD model: ${QUANT_DPD_MODEL_FILE}"
    fi
    # Use quantized model for run step
    FINAL_DPD_MODEL_FILE="${QUANT_DPD_MODEL_FILE}"
else
    # Use float model for run step
    FINAL_DPD_MODEL_FILE="${DPD_MODEL_FILE}"
fi

####################################################################################################
# 3. Validation Experiment
####################################################################################################
echo ""
echo "####################################################################################################"
echo "# Step: Run DPD                                                                                    #"
echo "####################################################################################################"
RUN_ARGS=(
    --dataset_name "${DATASET_NAME}"
    --step run_dpd
    --version "${VERSION_NAME}"
    --pretrained_model "${FINAL_DPD_MODEL_FILE}"
    --accelerator "${ACCELERATOR}"
    --devices "${DEVICES}"
    --seed "${SEED}"
    --PA_backbone "${PA_BACKBONE}"
    --PA_hidden_size "${PA_HIDDEN_SIZE}"
    --PA_num_layers "${PA_NUM_LAYERS}"
    --DPD_backbone "${DPD_BACKBONE}"
    --DPD_hidden_size "${DPD_HIDDEN_SIZE}"
    --DPD_num_layers "${DPD_NUM_LAYERS}"
    --frame_length "${FRAME_LENGTH}"
)

# Add quantization flags if enabled
if [ "${QUANT}" = "true" ]; then
    RUN_ARGS+=(
        --quant
        --n_bits_w "${QUANT_BITS_W}"
        --n_bits_a "${QUANT_BITS_A}"
    )
fi

"${PYTHON_BIN}" main.py "${RUN_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/run.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: DPD validation failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "All steps completed successfully!"
echo "=========================================="
echo "Version: ${VERSION_NAME}"
echo "Description: ${VERSION_DESC}"
echo "PA Model: ${PA_MODEL_FILE}"
echo "DPD Model: ${DPD_MODEL_FILE}"
if [ "${QUANT}" = "true" ]; then
    echo "Quantized DPD Model: ${QUANT_DPD_MODEL_FILE}"
fi
echo "Logs saved to: ${LOG_DIR}/"
if [ -f "${LOG_DIR}/pa.log" ]; then
    echo "  - pa.log: PA training log"
fi
if [ -f "${LOG_DIR}/dpd.log" ]; then
    echo "  - dpd.log: DPD training log"
fi
if [ "${QUANT}" = "true" ] && [ -f "${LOG_DIR}/dpd_quant.log" ]; then
    echo "  - dpd_quant.log: Quantized DPD training log"
fi
if [ -f "${LOG_DIR}/run.log" ]; then
    echo "  - run.log: DPD validation log"
fi
echo "=========================================="

