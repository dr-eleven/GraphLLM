#!/usr/bin/env bash
set -euo pipefail

# GraphSSR GRPO Stage 2: start from the Stage 1 Hugging Face export.
# All paths are hardcoded.

STAGE1_MODEL_PATH="/opt/data/private/GraphSSR-master/outputs/models/verl_checkpoints/1.5b/verl_grpo_graph_rl_stage1/global_step_816/actor/huggingface"
TRAIN_FILE="/opt/data/private/GraphSSR-master/outputs/synthetic_data/rl_data/train_9799_graphr2_subgraph.parquet"
VAL_FILE="/opt/data/private/GraphSSR-master/outputs/synthetic_data/rl_data/test_600_graphr2_subgraph.parquet"
RL_OUTPUT_DIR="/opt/data/private/GraphSSR-master/outputs/models/verl_checkpoints/1.5b/verl_grpo_graph_rl_stage2"
LOG_DIR="/opt/data/private/GraphSSR-master/logs"
# A normal invocation starts a fresh Stage 2 run from STAGE1_MODEL_PATH.
# Set STAGE2_RESUME_MODE=auto only when resuming this same Stage 2 run.
STAGE2_RESUME_MODE="${STAGE2_RESUME_MODE:-disable}"

ray stop --force 2>/dev/null || true

mkdir -p "${LOG_DIR}" "${RL_OUTPUT_DIR}"
cd /opt/data/private/GraphSSR-master/reinforcement_learning
echo "Starting Stage 2 GRPO from Stage 1 model: ${STAGE1_MODEL_PATH}"

# Conservative settings for full-parameter GRPO with a 1.5B/3B model on
# three 24 GiB RTX 4090 GPUs.
# vLLM sleep mode uses CuMemAllocator, which rejects expandable segments.
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
export VLLM_ATTENTION_BACKEND="XFORMERS"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="0"
export HF_ATTENTION_IMPLEMENTATION=flash_attention_2

python3 -m verl.trainer.main_ppo --config-name _generated_ppo_trainer \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=24 \
    data.val_batch_size=12 \
    data.max_prompt_length=4096 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${STAGE1_MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
    actor_rollout_ref.model.use_shm=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192  \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    data.dataloader_num_workers=0 \
    trainer.logger=console \
    trainer.project_name=verl_grpo_graph_rl \
    trainer.experiment_name=verl_grpo_graph_rl_stage2 \
    trainer.default_local_dir="${RL_OUTPUT_DIR}" \
    trainer.resume_mode="${STAGE2_RESUME_MODE}" \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=null \
    trainer.test_freq=50 \
    trainer.keep_best_and_last=true \
    trainer.best_checkpoint_metric=val-core/global/acc/mean@1 \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    +trainer.in_second_stage=true \
    +trainer.second_stage_reward_lambda=0.1 \
    trainer.total_epochs=2 2>&1 | tee -a verl_grpo_graph_rl_stage2.log

# cd GraphSSR-master/reinforcement_learning
# bash run_rl_stage2.sh 2>&1 | tee rl2.log
