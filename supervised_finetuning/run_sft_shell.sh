#!/usr/bin/env bash
set -euo pipefail

# Hard-coded local paths for this GraphSSR checkout.
PROJECT_DIR="/opt/data/private/GraphSSR-master"
SCRIPT_DIR="/opt/data/private/GraphSSR-master/supervised_finetuning"
CONDA_ENV="${CONDA_ENV:-/opt/data/private/conda/envs/graphssr}"
DATA_FILE="/opt/data/private/GraphSSR-master/outputs/synthetic_data/sft_data/graph_sft_data.json"
FILTERED_DATA_FILE="/opt/data/private/GraphSSR-master/outputs/synthetic_data/sft_data/graph_sft_data.filtered.full_sft.json"
FILTER_SCRIPT="/opt/data/private/GraphSSR-master/supervised_finetuning/filter_sft_by_token_length.py"
DATASET_INFO="/opt/data/private/GraphSSR-master/supervised_finetuning/dataset_info.json"
YAML_FILE="/opt/data/private/GraphSSR-master/supervised_finetuning/full_sft.yaml"

if ! "${CONDA_ENV}/bin/python" -c 'import llamafactory' >/dev/null 2>&1; then
    echo "ERROR: LlamaFactory is not importable from ${CONDA_ENV}/bin/python." >&2
    exit 1
fi

for required_file in "${DATA_FILE}" "${FILTER_SCRIPT}" "${DATASET_INFO}" "${YAML_FILE}"; do
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
# Reduce allocator fragmentation during long-sequence full fine-tuning.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${CONDA_ENV}/bin/python" "${FILTER_SCRIPT}" \
    --config "${YAML_FILE}" \
    --input "${DATA_FILE}" \
    --output "${FILTERED_DATA_FILE}"

GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
echo "Starting GraphSSR SFT with ${GPU_COUNT} GPU(s)."

LLAMAFACTORY_LAUNCHER="$(${CONDA_ENV}/bin/python -c 'import importlib.util; print(importlib.util.find_spec("llamafactory.launcher").origin)')"
if (( GPU_COUNT > 1 )); then
    MASTER_PORT="${MASTER_PORT:-$(${CONDA_ENV}/bin/python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')}"
    exec "${CONDA_ENV}/bin/python" -m torch.distributed.run --nnodes=1 --node-rank=0 \
        --nproc-per-node="${GPU_COUNT}" --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
        "${LLAMAFACTORY_LAUNCHER}" "${YAML_FILE}"
fi
exec "${CONDA_ENV}/bin/python" "${LLAMAFACTORY_LAUNCHER}" "${YAML_FILE}"
