#!/usr/bin/env bash
set -euo pipefail

# Evaluate the final GraphSSR Stage 1 model on all 600 RL validation samples.
# This follows Graph-R1's ds1.5b_eval_only.sh pattern: load a complete
# Hugging Face checkpoint directly, run the initial validation pass, then exit.

PROJECT_DIR="/opt/data/private/GraphSSR-master/reinforcement_learning"
MODEL_PATH="/opt/data/private/GraphSSR-master/outputs/models/verl_checkpoints/1.5b/verl_grpo_graph_rl_stage1/global_step_816/actor/huggingface"
TRAIN_FILE="/opt/data/private/GraphSSR-master/outputs/synthetic_data/rl_data/train_9799_graphr2_subgraph.parquet"
VAL_FILE="/opt/data/private/GraphSSR-master/outputs/synthetic_data/rl_data/test_600_graphr2_subgraph.parquet"
EVAL_OUTPUT_DIR="/opt/data/private/GraphSSR-master/outputs/test_result/1.5b/stage1_global_step_816/test_600"
GENERATION_DIR="${EVAL_OUTPUT_DIR}/generations"
LOG_PATH="${EVAL_OUTPUT_DIR}/eval.log"

mkdir -p "${EVAL_OUTPUT_DIR}" "${GENERATION_DIR}"
cd "${PROJECT_DIR}"

ray stop --force 2>/dev/null || true
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
export VLLM_ATTENTION_BACKEND="XFORMERS"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="0"
export HF_ATTENTION_IMPLEMENTATION=flash_attention_2

echo "Model:      ${MODEL_PATH}"
echo "Validation: ${VAL_FILE}"
echo "Log:        ${LOG_PATH}"

python3 -m verl.trainer.main_ppo --config-name _generated_ppo_trainer \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=24 \
    data.val_batch_size=12 \
    data.max_prompt_length=3072 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_shm=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=12 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.layered_summon=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    data.dataloader_num_workers=0 \
    trainer.logger=console \
    trainer.project_name=verl_grpo_graph_rl \
    trainer.experiment_name=verl_grpo_graph_rl_stage1_eval_test_600 \
    trainer.default_local_dir="${EVAL_OUTPUT_DIR}" \
    trainer.validation_data_dir="${GENERATION_DIR}" \
    trainer.resume_mode=disable \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    +trainer.in_second_stage=false \
    +trainer.second_stage_reward_lambda=0.1 \
    trainer.total_epochs=1 2>&1 | tee "${LOG_PATH}"

echo "Evaluation finished. Results: ${LOG_PATH}"

# conda activate graphssr1
#   cd /opt/data/private/GraphSSR-master/reinforcement_learning
# bash run_rl_stage1_eval.sh
