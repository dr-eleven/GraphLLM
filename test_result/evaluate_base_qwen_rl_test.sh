#!/usr/bin/env bash
set -euo pipefail

# Evaluate untouched Qwen2.5-Instruct base models on GraphSSR's 600-row RL
# validation/test set. No GraphSSR SFT adapter or merged SFT checkpoint is used.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2 bash test_result/evaluate_base_qwen_rl_test.sh all
#   CUDA_VISIBLE_DEVICES=0 bash test_result/evaluate_base_qwen_rl_test.sh 3b
#   CUDA_VISIBLE_DEVICES=0,1,2 bash test_result/evaluate_base_qwen_rl_test.sh 14b
#
# Optional overrides:
#   DATA_FILE=/path/to/test.parquet
#   PYTHON_BIN=/path/to/python
#   MAX_MODEL_LEN=16384 MAX_TOKENS=8192 BATCH_SIZE=8
#   QWEN14B_TP_SIZE=2 GPU_MEMORY_UTILIZATION=0.90

PROJECT_DIR="/opt/data/private/GraphSSR-master"
EVAL_SCRIPT="${PROJECT_DIR}/evaluation/evaluate_sft_ssr_600.py"
DATA_FILE="${DATA_FILE:-${PROJECT_DIR}/outputs/synthetic_data/rl_data/test_600_graphr2_subgraph.parquet}"
OUTPUT_ROOT="${PROJECT_DIR}/test_result"

QWEN3B_MODEL="/opt/data/private/Graph-R1-main/locals/Qwen/Qwen2.5-3B-Instruct"
QWEN14B_MODEL="/opt/data/private/Graph-R1-main/locals/Qwen/Qwen2.5-14B-Instruct"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
# Empty means: select the largest visible-GPU count that evenly divides the
# model's attention-head count. Qwen2.5-14B has 40 heads, so a three-GPU
# machine automatically selects TP=2 instead of the invalid TP=3.
QWEN14B_TP_SIZE="${QWEN14B_TP_SIZE:-}"
TARGET="${1:-all}"

find_python_with_vllm() {
    local candidate
    for candidate in \
        "/opt/data/private/.conda/envs/GraphLLM/bin/python" \
        "/opt/data/private/.conda/envs/graphssr/bin/python" \
        "/opt/data/private/conda/envs/graphssr1/bin/python" \
        "/opt/data/private/conda/envs/graphr1/bin/python" \
        "/opt/data/private/conda/envs/graphr1-trl/bin/python"; do
        if [[ -x "${candidate}" ]] && "${candidate}" -c \
            'import importlib.util; raise SystemExit(0 if all(importlib.util.find_spec(x) for x in ("torch", "transformers", "pandas", "vllm")) else 1)' \
            >/dev/null 2>&1; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

if [[ -n "${PYTHON_BIN:-}" ]]; then
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "ERROR: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
        exit 1
    fi
else
    if ! PYTHON_BIN="$(find_python_with_vllm)"; then
        echo "ERROR: no local Python environment containing torch, transformers, pandas, and vllm was found." >&2
        echo "Set PYTHON_BIN=/path/to/python and rerun." >&2
        exit 1
    fi
fi

case "${TARGET}" in
    all|3b|14b) ;;
    *)
        echo "Usage: $0 [all|3b|14b]" >&2
        exit 2
        ;;
esac

for required_file in \
    "${PYTHON_BIN}" \
    "${EVAL_SCRIPT}" \
    "${DATA_FILE}" \
    "${QWEN3B_MODEL}/config.json" \
    "${QWEN14B_MODEL}/config.json"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: required file not found: ${required_file}" >&2
        exit 1
    fi
done

if ! "${PYTHON_BIN}" -c \
    'import importlib.util; raise SystemExit(0 if all(importlib.util.find_spec(x) for x in ("torch", "transformers", "pandas", "vllm")) else 1)'; then
    echo "ERROR: ${PYTHON_BIN} does not contain all required packages: torch, transformers, pandas, vllm." >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "ERROR: CUDA is unavailable from ${PYTHON_BIN}. Check CUDA_VISIBLE_DEVICES and the NVIDIA driver." >&2
    exit 1
fi

VISIBLE_GPU_COUNT="$(${PYTHON_BIN} -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${TARGET}" != "3b" ]]; then
    QWEN14B_HEADS="$(${PYTHON_BIN} -c \
        'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["num_attention_heads"])' \
        "${QWEN14B_MODEL}/config.json")"

    if [[ -z "${QWEN14B_TP_SIZE}" ]]; then
        for (( candidate=VISIBLE_GPU_COUNT; candidate>=1; candidate-- )); do
            if (( QWEN14B_HEADS % candidate == 0 )); then
                QWEN14B_TP_SIZE="${candidate}"
                break
            fi
        done
    fi

    if [[ ! "${QWEN14B_TP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: QWEN14B_TP_SIZE must be a positive integer; got '${QWEN14B_TP_SIZE}'." >&2
        exit 1
    fi
    if (( QWEN14B_TP_SIZE > VISIBLE_GPU_COUNT )); then
        echo "ERROR: TP=${QWEN14B_TP_SIZE} needs that many visible GPUs; found ${VISIBLE_GPU_COUNT}." >&2
        exit 1
    fi
    if (( QWEN14B_HEADS % QWEN14B_TP_SIZE != 0 )); then
        echo "ERROR: Qwen2.5-14B has ${QWEN14B_HEADS} attention heads, which TP=${QWEN14B_TP_SIZE} cannot divide." >&2
        echo "Use QWEN14B_TP_SIZE=2 (recommended on this three-GPU host)." >&2
        exit 1
    fi

    echo "Qwen2.5-14B tensor parallel size: ${QWEN14B_TP_SIZE} (${QWEN14B_HEADS} attention heads)"
fi

# GraphSSR vendors veRL under reinforcement_learning/ but veRL itself uses
# absolute `verl.*` imports internally.  Keep both package roots visible:
#   reinforcement_learning.verl.* -> PROJECT_DIR
#   verl.*                        -> PROJECT_DIR/reinforcement_learning
export PYTHONPATH="${PROJECT_DIR}/reinforcement_learning:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Using Python environment: ${PYTHON_BIN}"

write_run_config() {
    local model_path="$1"
    local output_dir="$2"
    local tensor_parallel_size="$3"

    mkdir -p "${output_dir}"
    "${PYTHON_BIN}" - \
        "${model_path}" "${DATA_FILE}" "${output_dir}" \
        "${MAX_MODEL_LEN}" "${MAX_TOKENS}" "${BATCH_SIZE}" \
        "${tensor_parallel_size}" "${GPU_MEMORY_UTILIZATION}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

model, data, output, model_len, max_tokens, batch, tp, memory = sys.argv[1:]
config_path = pathlib.Path(model) / "config.json"
metadata = {
    "model_path": model,
    "model_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "model_weight_index_sha256": (
        hashlib.sha256((pathlib.Path(model) / "model.safetensors.index.json").read_bytes()).hexdigest()
        if (pathlib.Path(model) / "model.safetensors.index.json").is_file()
        else None
    ),
    "graphssr_sft_used": False,
    "data_path": data,
    "evaluation_script": "/opt/data/private/GraphSSR-master/evaluation/evaluate_sft_ssr_600.py",
    "decoding": "greedy",
    "max_model_len": int(model_len),
    "max_response_tokens": int(max_tokens),
    "batch_size": int(batch),
    "tensor_parallel_size": int(tp),
    "gpu_memory_utilization": float(memory),
    "started_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
}
(pathlib.Path(output) / "run_config.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

evaluate_model() {
    local model_path="$1"
    local output_dir="$2"
    local tensor_parallel_size="$3"

    write_run_config "${model_path}" "${output_dir}" "${tensor_parallel_size}"
    echo "Evaluating untouched base model: ${model_path}"
    echo "RL test data: ${DATA_FILE}"
    echo "Output: ${output_dir}"

    "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
        --model "${model_path}" \
        --data "${DATA_FILE}" \
        --output-dir "${output_dir}" \
        --batch-size "${BATCH_SIZE}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-tokens "${MAX_TOKENS}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --tensor-parallel-size "${tensor_parallel_size}"
}

mkdir -p "${OUTPUT_ROOT}"

if [[ "${TARGET}" == "all" || "${TARGET}" == "3b" ]]; then
    evaluate_model \
        "${QWEN3B_MODEL}" \
        "${OUTPUT_ROOT}/base_qwen2.5_3b_instruct" \
        1
fi

if [[ "${TARGET}" == "all" || "${TARGET}" == "14b" ]]; then
    evaluate_model \
        "${QWEN14B_MODEL}" \
        "${OUTPUT_ROOT}/base_qwen2.5_14b_instruct" \
        "${QWEN14B_TP_SIZE}"
fi

echo "Evaluation complete. Results are under ${OUTPUT_ROOT}."
