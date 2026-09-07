#!/usr/bin/env bash
set -euo pipefail

# GraphSSR Qwen2.5-14B LoRA SFT on three local 24 GiB GPUs.
PROJECT_DIR="/opt/data/private/GraphSSR-master"
SCRIPT_DIR="${PROJECT_DIR}/supervised_finetuning"
CONDA_ENV="${CONDA_ENV:-/opt/data/private/conda/envs/graphssr}"
MODEL_DIR="/opt/data/private/Graph-R1-main/locals/Qwen/Qwen2.5-14B-Instruct"
DATA_FILE="${PROJECT_DIR}/outputs/synthetic_data/sft_data/graph_sft_data.json"
# Keep a dedicated 14B filtered file: its shorter memory-safe cutoff must not
# overwrite the 3B training data.
FILTERED_DATA_FILE="${PROJECT_DIR}/outputs/synthetic_data/sft_data/graph_sft_data.filtered.lora_sft_14b.json"
FILTER_SCRIPT="${SCRIPT_DIR}/filter_sft_by_token_length.py"
DATASET_INFO="${SCRIPT_DIR}/dataset_info.json"
YAML_FILE="${SCRIPT_DIR}/lora_sft_14b.yaml"
DEEPSPEED_CONFIG="${SCRIPT_DIR}/ds_z3_offload_no_pin_config.json"

if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
    echo "ERROR: Python is not available in ${CONDA_ENV}." >&2
    exit 1
fi

for required_file in \
    "${MODEL_DIR}/config.json" \
    "${MODEL_DIR}/model.safetensors.index.json" \
    "${MODEL_DIR}/tokenizer.json" \
    "${DATA_FILE}" \
    "${FILTER_SCRIPT}" \
    "${DATASET_INFO}" \
    "${YAML_FILE}" \
    "${DEEPSPEED_CONFIG}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: required file not found: ${required_file}" >&2
        exit 1
    fi
done

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHON_EXEC="${CONDA_ENV}/bin/python"
export CUDA_HOME="${CONDA_ENV}"
export CUDA_PATH="${CUDA_HOME}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PROJECT_DIR}/.cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_DIR}/.cache/matplotlib}"
mkdir -p "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"

if ! "${CONDA_ENV}/bin/python" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "ERROR: CUDA is unavailable. Check the NVIDIA driver/GPU allocation, then rerun this script." >&2
    exit 1
fi

GPU_COUNT="$("${CONDA_ENV}/bin/python" -c 'import torch; print(torch.cuda.device_count())')"
if (( GPU_COUNT != 3 )); then
    echo "ERROR: this 14B configuration requires exactly 3 visible GPUs; found ${GPU_COUNT}." >&2
    echo "Set CUDA_VISIBLE_DEVICES to three 24 GiB GPUs, or retune gradient accumulation and memory settings." >&2
    exit 1
fi

if ! "${CONDA_ENV}/bin/python" -c \
    'import importlib.util; raise SystemExit(0 if all(importlib.util.find_spec(name) for name in ("flash_attn", "deepspeed", "llamafactory")) else 1)'; then
    echo "ERROR: flash-attn, DeepSpeed, or LlamaFactory is unavailable in ${CONDA_ENV}." >&2
    exit 1
fi

SFT_CUTOFF_LEN="$("${CONDA_ENV}/bin/python" -c \
    'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["cutoff_len"])' \
    "${YAML_FILE}")"
echo "LoRA 14B SFT complete-sequence cutoff_len=${SFT_CUTOFF_LEN} tokens."

"${CONDA_ENV}/bin/python" "${FILTER_SCRIPT}" \
    --config "${YAML_FILE}" \
    --input "${DATA_FILE}" \
    --output "${FILTERED_DATA_FILE}"

echo "Starting GraphSSR Qwen2.5-14B LoRA SFT with ${GPU_COUNT} GPUs (global batch=63)."

# This environment was relocated from /opt/conda, so console entry points such
# as llamafactory-cli and torchrun contain stale shebangs. Launch torchrun as a
# Python module to guarantee that the parent and every worker use CONDA_ENV.
LLAMAFACTORY_LAUNCHER="$("${CONDA_ENV}/bin/python" -c \
    'import importlib.util; print(importlib.util.find_spec("llamafactory.launcher").origin)')"
MASTER_PORT="${MASTER_PORT:-$("${CONDA_ENV}/bin/python" -c \
    'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')}"
exec "${CONDA_ENV}/bin/python" -m torch.distributed.run \
    --nnodes=1 \
    --node-rank=0 \
    --nproc-per-node="${GPU_COUNT}" \
    --master-addr=127.0.0.1 \
    --master-port="${MASTER_PORT}" \
    "${LLAMAFACTORY_LAUNCHER}" \
    "${YAML_FILE}"
