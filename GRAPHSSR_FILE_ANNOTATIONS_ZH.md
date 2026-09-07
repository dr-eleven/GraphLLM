# GraphSSR File Annotations

## Scope

This document complements the in-file comments added to Python, shell, and YAML files.

- For files that support comments, the main purpose and training-stage role were added directly in the file header.
- For files that cannot be safely commented, such as `JSON`, or for static assets and project documents, the explanation is recorded here.
- The `reinforcement_learning/` directory vendors a large upstream `verl` codebase. Below, the GraphSSR-specific training files are listed individually; the remaining upstream framework files are grouped by purpose to avoid treating third-party internals as GraphSSR-authored logic.

## Root Files

- `README.md`: Project entry document. Explains the end-to-end GraphSSR workflow, required datasets, model deployment, SFT, two-stage RL, and evaluation. It is the top-level operator guide for the whole training pipeline.
- `GRAPHSSR_PROJECT_GUIDE_ZH.md`: Chinese reading/running guide. Helps understand the repository structure and what each execution path is expected to do before attempting training.
- `LICENSE`: Project license. No direct role in training.
- `appendix/Online_Appendix.pdf`: Paper appendix. Research reference only; not executed during training.
- `.gitignore`: Git ignore rules. No direct role in training.

## Shared Infra and Utilities

- `scripts/deploy_api.sh`: Deploys teacher, SFT, or final evaluation models behind vLLM HTTP endpoints. Used before SFT data generation, RL data generation, and final evaluation.
- `scripts/start_ray_cluster.sh`: Recreates containers and starts the Ray head/workers. Required before Stage 1 and Stage 2 RL training.
- `utils/utils.py`: Shared batched HTTP inference helper. Reused by SFT teacher reasoning, RL difficulty estimation, and evaluation inference.

## Data Generation

- `data_generation/pipelines/prompt_templete.py`: Defines the canonical Sample-Select-Reason prompt and output schema. This prompt contract is shared by SFT data generation, RL data generation, reward parsing, and evaluation.
- `data_generation/pipelines/graph_prepare.py`: Normalizes GraphR1 examples and sampled subgraphs into prompt-ready graph fields. It feeds both the SFT and RL data construction stages.
- `data_generation/pipelines/sft_data_generator.py`: Full SFT data synthesis pipeline. Builds prompts, calls the teacher model, filters traces, and writes `graph_sft_data.json` for SFT training.
- `data_generation/pipelines/preprocess.py`: Samples local task-specific subgraphs with PyG and saves processed `.pt` files. This is the RL subgraph preparation stage before prompt generation.
- `data_generation/pipelines/rl_data_generator.py`: Builds RL train/validation prompts, estimates question difficulty with the SFT model, and exports Parquet files consumed by PPO/GRPO.
- `data_generation/quality_check/quality_evaluator.py`: Parses SSR outputs and performs authenticity, consistency, and answer checks. It is used during SFT filtering and mirrored by the RL reward logic.
- `data_generation/quality_check/graph_diversity.py`: Uses a diversity model to score pairwise subgraph differences. It is part of SFT data curation, not RL training.

## Supervised Fine-Tuning

- `supervised_finetuning/full_sft.yaml`: LlamaFactory training configuration for GraphSSR SFT. It controls model path, data source, output path, and training hyperparameters.
- `supervised_finetuning/run_sft_shell.sh`: Multi-node SFT launcher. Coordinates `prepare`, `start`, and `exec` actions across all SFT machines.
- `supervised_finetuning/run_sft_shell_node.sh`: Per-node SFT helper. Loads the image, starts the container, mounts data/config, and launches distributed LlamaFactory training.
- `supervised_finetuning/dataset_info.json`: LlamaFactory dataset registration file. It maps `graph_sft_data.json` fields to the prompt/response columns used in SFT. Because this is JSON, it was left syntax-safe and documented here instead of being commented inline.

## Evaluation

- `evaluation/evaluate_trained_model.py`: Rebuilds GOFA prompts into GraphSSR's SSR format, sends them to evaluation servers, parses answers, and computes accuracy/format metrics.
- `evaluation/run_evaluate.sh`: Launches the standard evaluation runs on the main GOFA set and the supplemental GOFA set.

## Reinforcement Learning: GraphSSR-Specific Files

- `reinforcement_learning/run_rl_stage1.sh`: Launches the first RL stage. This stage rewards SSR compliance and answer correctness without any subgraph-size bonus.
- `reinforcement_learning/run_rl_stage2.sh`: Launches the second RL stage from a Stage 1 checkpoint. This stage keeps the Stage 1 reward and adds the denoising size preference.
- `reinforcement_learning/verl/trainer/config/ppo_trainer.yaml`: Shared PPO/GRPO trainer config extended with `in_second_stage` and `second_stage_reward_lambda` so one trainer can serve both RL stages.
- `reinforcement_learning/verl/trainer/ppo/ray_trainer.py`: Main distributed training loop. GraphSSR relies on this file to push the stage flags into training and validation batches.
- `reinforcement_learning/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`: Rollout-side sequence generation. It produces the sampled SSR responses that are later parsed and scored by the GraphSSR reward.
- `reinforcement_learning/verl/workers/reward_manager/naive.py`: Reward-manager bridge from generic verl batches to GraphSSR reward scoring. It injects prompt text and stage metadata into reward computation.
- `reinforcement_learning/verl/utils/reward_score/__init__.py`: Reward-function dispatch registry. It routes `graph_rl_data` samples to GraphSSR's custom reward implementation.
- `reinforcement_learning/verl/utils/reward_score/subgraph_size.py`: Core GraphSSR reward function. Stage 1 scores authenticity, consistency, and answer correctness; Stage 2 adds a size-based denoising bonus.

## Reinforcement Learning: Upstream veRL Files Present in This Repo

- `reinforcement_learning/README.md`: Upstream veRL usage documentation. Useful for environment setup, but not the GraphSSR-specific training logic.
- `reinforcement_learning/setup.py` and `reinforcement_learning/pyproject.toml`: veRL package installation metadata. Used when preparing the RL runtime environment.
- `reinforcement_learning/requirements.txt`, `requirements-cuda.txt`, `requirements-npu.txt`, `requirements_sglang.txt`, `requirements_transferqueue.txt`: Dependency manifests for different RL runtime variants.
- `reinforcement_learning/LICENSE` and `reinforcement_learning/Notice.txt`: Upstream legal files. No direct role in training.
- `reinforcement_learning/.gitignore`, `.pre-commit-config.yaml`, `.readthedocs.yaml`, `CONTRIBUTING.md`: Repository maintenance files. No direct role in GraphSSR training.
- `reinforcement_learning/docs/`, `examples/`, `tests/`: Upstream reference, example, and test material. Helpful for debugging or extending veRL, but not part of the default GraphSSR training path.
- `reinforcement_learning/recipe/`: Optional upstream recipe directory. GraphSSR's default training path in this repo does not launch these recipes directly.
- `reinforcement_learning/verl/` other than the files listed above: Core veRL framework internals that provide distributed training, rollout, checkpointing, and worker abstractions for GraphSSR's RL stage.
