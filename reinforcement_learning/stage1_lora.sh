#!/usr/bin/env bash
set -euo pipefail

# GraphSSR GRPO Stage 1: initialize from the merged SFT model and train a new
# LoRA adapter with the original GraphSSR reward.

SFT_MODEL_PATH="/opt/data/private/GraphSSR-master/outputs/sft/ds1.5b_lora_merged"
TRAIN_FILE="/opt/data/private/GraphSSR-master/outputs/synthetic_data/rl_data/train_9799_graphr2_subgraph_strict_ssr.parquet"
VAL_FILE="/opt/data/private/GraphSSR-master/outputs/synthetic_data/rl_data/test_600_graphr2_subgraph_strict_ssr.parquet"
RL_OUTPUT_DIR="/opt/data/private/GraphSSR-master/outputs/rl/1.5b_lora/verl_grpo_graph_rl_stage1_lora"
LOG_DIR="/opt/data/private/GraphSSR-master/logs"
RUN_LOG="${LOG_DIR}/verl_grpo_graph_rl_stage1_lora.log"
RL_ENV_BIN="${RL_ENV_BIN:-/opt/conda/envs/graphssr1/bin}"
RL_PYTHON="${RL_ENV_BIN}/python"
RAY_CLI="${RL_ENV_BIN}/ray"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
RL_LEARNING_RATE="${RL_LEARNING_RATE:-3e-6}"

if [[ ! -x "${RL_PYTHON}" || ! -x "${RAY_CLI}" ]]; then
    echo "RL environment is incomplete under ${RL_ENV_BIN}" >&2
    exit 1
fi

if [[ ! -f "${SFT_MODEL_PATH}/config.json" || ! -f "${SFT_MODEL_PATH}/model.safetensors" ]]; then
    echo "Merged SFT model is incomplete under ${SFT_MODEL_PATH}." >&2
    echo "Run supervised_finetuning/merge.yaml before starting RL training." >&2
    exit 1
fi

"${RAY_CLI}" stop --force 2>/dev/null || true

mkdir -p "${LOG_DIR}" "${RL_OUTPUT_DIR}"
cd /opt/data/private/GraphSSR-master/reinforcement_learning
echo "Starting Stage 1 GRPO LoRA from merged SFT model: ${SFT_MODEL_PATH}"
echo "LoRA rank=${LORA_RANK}, alpha=${LORA_ALPHA}, learning_rate=${RL_LEARNING_RATE}"

# Conservative settings for LoRA GRPO with a 1.5B model on three 24 GiB
# RTX 4090 GPUs. The merged SFT weights stay frozen; only this new RL LoRA
# adapter is optimized.
# vLLM sleep mode uses CuMemAllocator, which rejects expandable segments.
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
# export VLLM_ATTENTION_BACKEND="XFORMERS"
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
    actor_rollout_ref.model.path="${SFT_MODEL_PATH}" \
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
    trainer.experiment_name=verl_grpo_graph_rl_stage1_lora \
    trainer.default_local_dir="${RL_OUTPUT_DIR}" \
    trainer.resume_mode=auto \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=null \
    trainer.test_freq=100 \
    trainer.val_before_train=False \
    trainer.keep_best_and_last=true \
    trainer.best_checkpoint_metric=val-core/global/acc/mean@1 \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    +trainer.in_second_stage=false \
    +trainer.second_stage_reward_lambda=0.1 \
    trainer.total_epochs=2 2>&1 | tee -a "${RUN_LOG}"


# cd /opt/data/private/GraphSSR-master/reinforcement_learning
# bash stage1_lora.sh 2>&1 | tee rl_lora.log
