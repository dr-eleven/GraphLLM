# GraphSSR-master 与 GraphSSR-master-ori 差异说明

## 1. 对比范围与方法

本文件比较：

- 修改版：`/opt/data/private/GraphSSR-master`
- 原版：`/opt/data/private/GraphSSR-master-ori`

对比使用文件清单和逐文件内容 diff；`.git`、Python 缓存和运行时生成物不作为源代码功能差异解读。修改版中的新增日志、模型、数据、报告和缓存目录单独列在文末。

总体上，修改版不仅增加了中文注释和项目说明，还加入了 Graph-R1 数据兼容、严格 SSR 数据质量检查、长序列 SFT/RL 显存修复、断点/最佳 checkpoint 管理、可复现实验脚本和本地模型路径配置。

## 2. 文件清单差异

### 原版中存在、修改版仍保留但内容发生变化的主要文件

`data_generation/pipelines/graph_prepare.py`、`preprocess.py`、`prompt_templete.py`、`rl_data_generator.py`、`sft_data_generator.py`、`data_generation/quality_check/graph_diversity.py`、`evaluation/evaluate_trained_model.py`、`evaluation/run_evaluate.sh`、`reinforcement_learning/run_rl_stage1.sh`、`run_rl_stage2.sh`、`reinforcement_learning/verl/trainer/ppo/ray_trainer.py`、`verl/utils/fsdp_utils.py`、`verl/utils/reward_score/subgraph_size.py`、`verl/workers/config/rollout.py`、`verl/workers/fsdp_workers.py`、`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`、`scripts/deploy_api.sh`、`scripts/start_ray_cluster.sh`、`supervised_finetuning/dataset_info.json`、`full_sft.yaml`、`run_sft_shell.sh`、`run_sft_shell_node.sh`、`utils/utils.py`。

### 修改版新增的主要源代码、配置和文档

- 数据/评测：`data_generation/pipelines/rebuild_graph_bert_encoded.py`、`evaluation/evaluate_sft_ssr_600.py`、`evaluation/evaluate_trained_model_note.py`。
- 训练启动：`reinforcement_learning/run_rl_stage1_eval.sh`、`stage1_lora.sh`、`stage2_lora.sh`、`reinforcement_learning/verl/trainer/main_ppo_note.py`。
- 训练测试工具：`reinforcement_learning/tests/trainer/ppo/test_best_checkpoint_retention_on_cpu.py`、`reinforcement_learning/tools/`、`scripts/download_gofa_test_pt.sh`、`scripts/upgrade_rl_prompts.py`。
- SFT 配置和工具：`full_sft_3b.yaml`、`lora_sft_1.5b.yaml`、`lora_sft_3b.yaml`、`lora_sft_14b.yaml`、`merge.yaml`、`merge_lora_3b.yaml`、`merge_lora_14b.yaml`、多个 DeepSpeed 配置、`filter_sft_by_token_length.py` 及对应启动脚本。
- 中文项目资料：`GRAPHSSR_FILE_ANNOTATIONS_ZH.md`、`GRAPHSSR_PROJECT_GUIDE_ZH.md`、`GRAPHSSR_REPRODUCTION_GUIDE_ZH.md`、`teacher.md`、`loss.md`。

## 3. 数据生成与 prompt 处理

### `graph_prepare.py`

- 增加 Graph-R1 SFT/RL 数据字段解析和标准化说明。
- 对 SFT prompt 提取目标节点、节点文本、边、标签列表和答案；对 RL Parquet prompt 提取结构化图信息并检查缺失字段。
- 增加/明确数据集与任务配额（`nc`、`lp`、`lc`，以及 easy/medium/hard 难度）。
- 增加从 Graph-R1 SFT 数据和子图池生成图信息缓存的流程。
- 主要是数据格式兼容、采样配额和可恢复缓存逻辑；大量改动是中文注释和流程说明。

### `preprocess.py`

- 明确使用 PyG `NeighborLoader`、`LinkNeighborLoader` 和 `RandomLinkSplit` 处理节点分类、链接预测、链接分类。
- 规定不同任务的邻居采样规模；对过小/过大的子图进行过滤。
- 对子图节点进行随机重排并同步重映射 `edge_index`，降低位置偏差。
- 修改版强化了数据预处理行为的说明，核心采样逻辑与原版总体一致。

### `prompt_templete.py`

- 在 SSR prompt 中增加更明确的结构化输出约束。
- 子图采样阶段要求恰好生成指定数量、连续编号的完整子图块。
- `answer_format_subgraph_sampling_prompt()` 改为按 `group_size` 动态展开多个 `<Subgraph_i>` 块，并补充中心节点、邻居、连接关系和采样理由的字段规范。
- 推理、选择和答案格式的模板说明得到扩展，便于与后续质量检查和奖励解析器对齐。

### `rl_data_generator.py`

- 保留原有的多次教师模型推理和 easy/medium/hard 难度划分，但增加断点续传、已采样索引和已检查样本状态的持久化说明。
- 增强 RL 数据到 Parquet 的转换和测试数据转换流程。
- 明确通过多个 vLLM API 端点执行推理，并按样本组织多次尝试结果。

### `sft_data_generator.py` 与 `graph_diversity.py`

- SFT 数据生成流程增加教师推理、结果排序、已生成样本恢复和增量处理。
- SFT 过滤从少量原始检查扩展为严格 SSR 质量门槛：子图数量/编号、候选格式、真实子图、中心节点-only、结构多样性、选中子图格式与一致性、答案正确性。
- LLM 多样性评分只对通过结构性检查的样本执行，阈值仍为 `> 0.3`。
- 多样性评分模块增加批量请求、已有评分加载、增量追加保存和更明确的子图差异 prompt。

## 4. 评测流程

### `evaluation/evaluate_trained_model.py`

- 修改版使用基于文件位置的 `ROOT_DIR`，避免从不同当前工作目录启动时相对路径失效。
- 增加 `ensure_eval_graph_pt()`：优先寻找已有图文件，缺失时调用 `rebuild_graph_bert_encoded.py` 生成 products、wikics、fb15k237 等评测图。
- 评测 prompt 重建、答案集合加载和输出路径改为基于仓库根目录的可复现路径。
- 增强答案状态分类（正确、错误、未知、格式错误）和指标输出。
- API 推理增加最多 8 次的格式合规重试逻辑。

### 新增评测入口

- `evaluate_sft_ssr_600.py`：面向 600 条 SFT/SSR 数据的独立评测入口。
- `evaluate_trained_model_note.py`：带有详细流程说明、prompt 重建、答案解析和指标计算的注释版评测脚本。
- `run_evaluate.sh`：从原版的简短调用脚本扩展为参数检查、目录准备、输出文件说明和可审计运行流程。

## 5. SFT 配置和启动方式

### 配置变化

- 原版配置使用占位模型、数据集和输出路径；修改版改为本地绝对路径，例如 `/opt/data/private/GraphSSR-master` 和 `Graph-R1-main/locals` 下的模型。
- 增加 `full_sft_3b.yaml`，并增加 1.5B/3B/14B LoRA SFT 配置及对应 LoRA 合并配置。
- 使用 ZeRO-3 offload 配置、BF16、长序列 cutoff、梯度检查点和更适合 24 GiB GPU 的 batch/梯度累积设置。
- 新增 `filter_sft_by_token_length.py`，在 SFT 前按 tokenizer 真实计算完整 prompt+response 长度，生成过滤缓存并保存元数据。
- 增加多份 DeepSpeed offload/no-pin 配置，以适配长序列和有限显存环境。

### `run_sft_shell.sh`

- 原版依赖多节点 SSH、Docker 和远程脚本；修改版改为单机本地执行 `llamafactory-cli train`。
- 增加输入文件、环境、CUDA、数据集配置检查。
- 自动执行 token 长度过滤，按 GPU 数决定是否启用 TorchRun，并设置 tokenizers、allocator 和 CUDA 相关环境变量。
- 增加断点恢复和本地日志/输出目录管理。

## 6. RL/verl 训练变化

### `run_rl_stage1.sh` 与 `run_rl_stage2.sh`

- 从原版的集群占位路径改为本地 GraphSSR 路径和环境检查。
- 增加 Ray 强制清理、日志目录、Ray CLI/Python 可执行文件检查和失败即退出。
- 针对 1.5B/3B 与三张 24 GiB GPU 调整 batch、prompt/response 长度、动态 batch、mini-batch、学习率、KL 系数和模型 dtype。
- 增加 `keep_best_and_last`、最佳指标、恢复模式、checkpoint 保留和训练 epoch 配置。
- 处理 vLLM sleep mode、XFORMERS、attention 实现及 CUDA allocator 兼容性。

### `verl/trainer/ppo/ray_trainer.py`

这是修改版最重要的功能差异之一：

- 增加第二阶段 reward 参数接入，使同一 trainer 支持两个 RL 阶段。
- 将验证分组从单一 `data_source` 扩展为 `dataset+task`，并新增全局 micro-average 指标。
- 将 GraphSSR reward 中的 `pred` 映射为 `acc`，使其显示为核心验证准确率。
- 增加最佳 checkpoint 指标、最佳/最近 checkpoint 保留、断点恢复状态文件和 checkpoint 存在性检查。
- 当启用 `keep_best_and_last` 时关闭 worker 的简单 recency rotation，避免保存新 checkpoint 时删除旧的最佳 checkpoint。
- 增加 checkpoint 清理、最佳状态保存和恢复相关逻辑。

### FSDP/vLLM 稳定性修复

- `verl/utils/fsdp_utils.py`：参数和 optimizer state 的 CPU/GPU offload 改为同步 copy，避免异步传输尚未完成就被释放或被下一阶段使用。
- `verl/workers/fsdp_workers.py`：reference policy 的 CPU offload 改为尊重 `param_offload` 配置；rollout 构建 vLLM 前强制同步并清理缓存，避免 profiling 内存基线异常。
- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`、rollout 配置和相关 YAML 增加 GraphSSR reward/多样本训练兼容项。
- `subgraph_size.py` 的功能基本未改变，主要是文件末尾换行和注释差异。

## 7. 推理服务、Ray 和共享工具

### `scripts/deploy_api.sh`

- 原版是占位部署脚本；修改版增加完整的 Docker/vLLM/SSH 多节点部署说明和流程。
- 支持镜像检查/加载、旧容器清理、`--gpus all`、host network、共享 IPC、vLLM 服务启动和日志。
- 明确 `MODEL_PATH`、`TP_SIZE`、端口和 `SERVED_MODEL_NAME` 必须与调用方一致，并区分 teacher、diversity、SFT 和 eval 服务。

### `scripts/start_ray_cluster.sh`

- 增加多节点 Docker 容器初始化、Ray head/worker 启动、节点存活检查和失败重试。
- 将第一个节点固定为 head，使用 host network 和共享内存配置。

### `utils/utils.py`

- 保留多 endpoint HTTP 推理、线程池、重试和结果排序功能。
- 增加请求响应解析、completion token 统计、吞吐量监控、任务失败重试和更详细的状态输出说明。
- 主要功能仍是调用 OpenAI 兼容的 vLLM API；修改版将其包装成更易恢复和审计的批处理调度器。

## 8. 路径、数据集和模型资源变化

- 修改版大量使用本机绝对路径，适合当前 `/opt/data/private` 环境，但降低了直接复制到其他机器后的可移植性。
- `dataset_info.json` 从原版的单一占位数据集扩展为 full SFT、1.5B/3B/14B LoRA 等多种数据集映射。
- 修改版新增 `data/`、`locals/`、训练输出和多份实验结果，包含 Graph-R1 数据、Qwen/DeepSeek 模型、SFT/RL checkpoint、预测结果和报告。

## 9. 非功能性差异

- 大量 Python 文件增加中文 docstring、中文注释、类型/流程说明和更明确的错误信息。
- `.gitignore` 增加对当前实验输出、缓存和生成文件的忽略规则。
- 修改版包含 `.cache`、`.pytest_cache`、`verl.egg-info`、日志、报告、预测和模型输出等运行时产物；这些不是原始算法实现差异。
- `reinforcement_learning/verl/utils/reward_score/subgraph_size.py` 的实质逻辑未见变化，主要是末尾换行。

## 10. 主要影响总结

修改版相对原版的实际工程方向可以概括为：

1. 将原版的占位路径/多节点模板改造成当前机器上的可运行本地流程。
2. 接入 Graph-R1 的数据结构、SSR prompt 和更严格的 SFT 质量过滤。
3. 增加 Qwen/DeepSeek 多模型、1.5B/3B/14B SFT/LoRA 配置和合并流程。
4. 针对 24 GiB GPU 的长序列训练、FSDP offload、vLLM profiling 和 CUDA allocator 做稳定性修复。
5. 增加按数据集/任务的验证指标、全局准确率、最佳 checkpoint 保留和断点恢复能力。
6. 增加可审计的评测、数据重建、prompt 升级、部署和实验报告工具。

## 11. 需要注意的兼容性风险

- 修改版脚本中的绝对路径需要在其他机器上重新配置。
- 修改版依赖的本地模型、数据文件、DeepSpeed 配置和 vLLM/Ray 环境，在原版中并不存在。
- 更严格的 SSR 质量检查会减少可用于 SFT 的样本数量，并可能改变训练数据分布。
- 新增的验证分组和 `acc` 指标改变了日志字段命名；旧的实验分析脚本可能只认识原版 `data_source` 指标。
- 修改版的 prompt 输出约束更严格，必须与 reward parser、SFT 数据格式和评测解析器同时使用，不能只替换单个 prompt 文件。
