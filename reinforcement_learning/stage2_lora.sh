#!/usr/bin/env bash
set -euo pipefail

# GraphSSR GRPO Stage 2 LoRA.
#
# The rollout/optimization settings intentionally match stage1_lora.sh.  Only
# the stage-2 inputs, output/log names, and second-stage reward switch differ.
# By default, start from stage1's best checkpoint (global_step_800); override
# STAGE1_STEP=814 to start from the final checkpoint instead.

GRAPHSSR_ROOT="/opt/data/private/GraphSSR-master"
STAGE1_STEP="${STAGE1_STEP:-800}"
STAGE1_MODEL_PATH="${STAGE1_MODEL_PATH:-${GRAPHSSR_ROOT}/outputs/rl/1.5b_lora/verl_grpo_graph_rl_stage1_lora/global_step_${STAGE1_STEP}/actor/huggingface}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${GRAPHSSR_ROOT}/outputs/sft/ds1.5b_lora_merged}"
MERGED_STAGE1_MODEL_PATH="${MERGED_STAGE1_MODEL_PATH:-${GRAPHSSR_ROOT}/outputs/rl/1.5b_lora/merged_stage1_step_${STAGE1_STEP}}"

# Stage 2 keeps its original (non-strict-SSR) reward data.  Set these variables
# explicitly if a different stage-2 dataset is intended.
TRAIN_FILE="/opt/data/private/GraphSSR-master/outputs/synthetic_data/rl_data/train_9799_graphr2_subgraph_strict_ssr.parquet"
VAL_FILE="/opt/data/private/GraphSSR-master/outputs/synthetic_data/rl_data/test_600_graphr2_subgraph_strict_ssr.parquet"
RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-${GRAPHSSR_ROOT}/outputs/rl/1.5b_lora/verl_grpo_graph_rl_stage2_lora}"
LOG_DIR="${LOG_DIR:-${GRAPHSSR_ROOT}/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/verl_grpo_graph_rl_stage2_lora.log}"

RL_ENV_BIN="${RL_ENV_BIN:-/opt/conda/envs/graphssr1/bin}"
RL_PYTHON="${RL_ENV_BIN}/python"
RAY_CLI="${RL_ENV_BIN}/ray"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
RL_LEARNING_RATE="${RL_LEARNING_RATE:-3e-6}"
STAGE2_RESUME_MODE="${STAGE2_RESUME_MODE:-disable}"

if [[ ! -x "${RL_PYTHON}" || ! -x "${RAY_CLI}" ]]; then
    echo "RL environment is incomplete under ${RL_ENV_BIN}" >&2
    exit 1
fi
if [[ ! -f "${STAGE1_MODEL_PATH}/config.json" || ! -f "${STAGE1_MODEL_PATH}/model.safetensors" ]]; then
    echo "Stage 1 Hugging Face export is incomplete: ${STAGE1_MODEL_PATH}" >&2
    echo "Set STAGE1_STEP or STAGE1_MODEL_PATH to a valid checkpoint." >&2
    exit 1
fi
if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
    echo "Stage 2 data file is missing." >&2
    exit 1
fi

# verl's hf_model export is a PEFT-wrapped state dict (base_layer/lora_A/lora_B),
# not a loadable plain Qwen checkpoint. Merge it before handing it to actor/vllm.
if [[ ! -f "${MERGED_STAGE1_MODEL_PATH}/model.safetensors" && ! -f "${MERGED_STAGE1_MODEL_PATH}/model-00001-of-00001.safetensors" ]]; then
    echo "Merging Stage 1 LoRA checkpoint into: ${MERGED_STAGE1_MODEL_PATH}"
    "${RL_PYTHON}" "${GRAPHSSR_ROOT}/reinforcement_learning/tools/merge_stage1_lora.py" \
        --base-model "${BASE_MODEL_PATH}" \
        --lora-export "${STAGE1_MODEL_PATH}" \
        --output "${MERGED_STAGE1_MODEL_PATH}" \
        --lora-alpha "${LORA_ALPHA}"
fi

"${RAY_CLI}" stop --force 2>/dev/null || true
mkdir -p "${LOG_DIR}" "${RL_OUTPUT_DIR}"
cd "${GRAPHSSR_ROOT}/reinforcement_learning"

echo "Starting Stage 2 GRPO LoRA from merged model: ${MERGED_STAGE1_MODEL_PATH}"
echo "Stage 1 LoRA source checkpoint: ${STAGE1_MODEL_PATH}"
echo "LoRA rank=${LORA_RANK}, alpha=${LORA_ALPHA}, learning_rate=${RL_LEARNING_RATE}"
echo "Stage 2 train data: ${TRAIN_FILE}"

# These runtime settings are shared with stage1_lora.sh.
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="0"
export HF_ATTENTION_IMPLEMENTATION=flash_attention_2

"${RL_PYTHON}" -m verl.trainer.main_ppo --config-name _generated_ppo_trainer \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=24 \
    data.val_batch_size=12 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.temperature=0.5 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.repetition_penalty=1.03 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MERGED_STAGE1_MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
    actor_rollout_ref.model.target_modules=all-linear \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    actor_rollout_ref.actor.optim.lr="${RL_LEARNING_RATE}" \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
    actor_rollout_ref.model.use_shm=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}" \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.cudagraph_capture_sizes="[1,2,4,8,16,24,32]" \
    actor_rollout_ref.rollout.max_model_len=12288 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    data.dataloader_num_workers=0 \
    trainer.logger=console \
    trainer.project_name=verl_grpo_graph_rl \
    trainer.experiment_name=verl_grpo_graph_rl_stage2_lora \
    trainer.default_local_dir="${RL_OUTPUT_DIR}" \
    trainer.resume_mode="${STAGE2_RESUME_MODE}" \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=null \
    trainer.test_freq=100 \
    trainer.val_before_train=False \
    trainer.keep_best_and_last=true \
    trainer.best_checkpoint_metric=val-core/global/acc/mean@1 \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    +trainer.in_second_stage=true \
    +trainer.second_stage_reward_lambda=0.1 \
    trainer.total_epochs=2 2>&1 | tee -a "${RUN_LOG}"

# bash stage2_lora.sh 2>&1 | tee rl2_lora.log
