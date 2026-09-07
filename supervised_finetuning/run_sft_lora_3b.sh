#!/usr/bin/env bash
set -euo pipefail

# GraphSSR Qwen2.5-3B LoRA SFT on the local three-GPU machine.
PROJECT_DIR="/opt/data/private/GraphSSR-hz"
SCRIPT_DIR="${PROJECT_DIR}/supervised_finetuning"
CONDA_ENV="${CONDA_ENV:-/opt/data/private/conda/envs/graphssr}"
MODEL_DIR="/opt/data/private/Graph-R1-main/locals/Qwen/Qwen2.5-3B-Instruct"
DATA_FILE="${PROJECT_DIR}/outputs/synthetic_data/sft_data/graph_sft_data.json"
FILTERED_DATA_FILE="${PROJECT_DIR}/outputs/synthetic_data/sft_data/graph_sft_data.filtered.lora_sft_3b.json"
FILTER_SCRIPT="${SCRIPT_DIR}/filter_sft_by_token_length.py"
DATASET_INFO="${SCRIPT_DIR}/dataset_info.json"
YAML_FILE="${SCRIPT_DIR}/lora_sft_3b.yaml"

if ! "${CONDA_ENV}/bin/python" -c 'import llamafactory' >/dev/null 2>&1; then
    echo "ERROR: LlamaFactory is not importable from ${CONDA_ENV}/bin/python." >&2
    exit 1
fi

for required_file in \
    "${MODEL_DIR}/config.json" \
    "${MODEL_DIR}/model.safetensors.index.json" \
    "${MODEL_DIR}/tokenizer.json" \
    "${DATA_FILE}" \
    "${FILTER_SCRIPT}" \
    "${DATASET_INFO}" \
    "${YAML_FILE}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: required file not found: ${required_file}" >&2
        exit 1
    fi
done

if ! "${CONDA_ENV}/bin/python" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "ERROR: CUDA is unavailable. Check the NVIDIA driver/GPU allocation, then rerun this script." >&2
    exit 1
fi

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SFT_CUTOFF_LEN="$("${CONDA_ENV}/bin/python" -c \
    'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["cutoff_len"])' \
    "${YAML_FILE}")"
echo "LoRA 3B SFT complete-sequence cutoff_len=${SFT_CUTOFF_LEN} tokens."

"${CONDA_ENV}/bin/python" "${FILTER_SCRIPT}" \
    --config "${YAML_FILE}" \
    --input "${DATA_FILE}" \
    --output "${FILTERED_DATA_FILE}"

GPU_COUNT="$("${CONDA_ENV}/bin/python" -c 'import torch; print(torch.cuda.device_count())')"
if (( GPU_COUNT < 1 )); then
    echo "ERROR: no visible GPU was found." >&2
    exit 1
fi

echo "Starting GraphSSR Qwen2.5-3B LoRA SFT with ${GPU_COUNT} GPU(s)."
if (( GPU_COUNT != 3 )); then
    echo "WARNING: gradient_accumulation_steps=21 gives global batch $((GPU_COUNT * 21)); the target is approximately 64." >&2
fi

LLAMAFACTORY_LAUNCHER="$(${CONDA_ENV}/bin/python -c 'import importlib.util; print(importlib.util.find_spec("llamafactory.launcher").origin)')"
if (( GPU_COUNT > 1 )); then
    MASTER_PORT="${MASTER_PORT:-$(${CONDA_ENV}/bin/python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')}"
    exec "${CONDA_ENV}/bin/python" -m torch.distributed.run --nnodes=1 --node-rank=0 \
        --nproc-per-node="${GPU_COUNT}" --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
        "${LLAMAFACTORY_LAUNCHER}" "${YAML_FILE}"
fi
exec "${CONDA_ENV}/bin/python" "${LLAMAFACTORY_LAUNCHER}" "${YAML_FILE}"
