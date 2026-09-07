# GraphSSR

源代码网站：【https://github.com/mysteriouslfz/GraphSSR】

## 1. 环境配置
我这里提供我用的环境设置

```bash
conda create -n graphssr python=3.10 -y
conda activate graphssr

# 先装 PyTorch（CUDA 12.4）
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 安装环境，我提供了我的环境
pip install -r graphssr.txt --no-deps

# flash-attn，需要额外下载，我是用whl装的
pip install flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
这样应该就行了，不行的让llm来装吧。。

## 2. 数据和模型下载

数据可从 [Hugging Face GraphSSR 数据集](https://huggingface.co/datasets/wfdlfz/GraphSSR/tree/main) 下载。当前代码期望：

```text
data/train_data/raw_data/graphr1/sft_train_data_filter2048_10000_balance.csv
data/train_data/raw_data/{dataset}/{dataset}_summary.jsonl
data/train_data/raw_data/{dataset}/{dataset}_graph_bert_encoded.pt
data/train_data/rl_data/test_1050.parquet
data/test_data/raw_data/gofa_test_data/gofa_test_data_53114.csv
data/test_data/raw_data/gofa_test_data/gofa_supply_data_35603.csv
```

RL 子图生成使用 `arxiv`、`children`、`computer`、`history`、`photo`、`pubmed`、`reddit`、`sports`、`wn18rr`；Graph-R1 兼容评测还使用 `cora_node`、`products`、`wikics`、`fb15k237`。

缺少 Graph BERT 文件时：

```bash
cd /opt/data/private/GraphSSR-master
python data_generation/pipelines/rebuild_graph_bert_encoded.py \
  --datasets products wikics fb15k237 \
  --output-root data/train_data/raw_data
```

训练模型：

```text
/opt/data/private/Graph-R1-main/locals/DeepSeek-R1-Distill-Qwen-14B
```
下载
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --local-dir /opt/data/private/GraphSSR-master/locals/DeepSeek-R1-Distill-Qwen-14B --exclude "*.DS_Store" "imgs/*"

生成数据默认位于：

```text
outputs/synthetic_data/sft_data/graph_sft_data.json
outputs/synthetic_data/rl_data/train_9799_graphr2_subgraph.parquet
outputs/synthetic_data/rl_data/test_600_graphr2_subgraph.parquet
```

## 3. SFT 训练

### 3.1 生成 SFT 数据

```bash
cd /opt/data/private/GraphSSR-master/data_generation/pipelines
python sft_data_generator.py
```

流程包括 Graph-R1 图信息提取、SSR prompt 构造、教师 vLLM 推理、结构多样性计算，以及子图数量/真实性/中心节点/选择一致性/答案正确性检查。教师 endpoint 在 `utils/utils.py` 中配置。

### 3.2 Full SFT

配置文件：

```text
supervised_finetuning/full_sft.yaml
supervised_finetuning/full_sft_3b.yaml
```

确认 `model_name_or_path`、`dataset`、`dataset_dir`、`deepspeed`、`output_dir` 和 `cutoff_len`，然后运行：

```bash
cd /opt/data/private/GraphSSR-master/supervised_finetuning
bash run_sft_shell.sh
# 3B 专用：
bash run_sft_3b.sh
```

脚本会先按 tokenizer 过滤完整 prompt+response 长度，再通过所选环境的 Python 模块调用 `llamafactory` 进行训练。

### 3.3 LoRA SFT

```text
lora_sft_1.5b.yaml  -> run_sft_lora_1.5b.sh
lora_sft_3b.yaml    -> run_sft_lora_3b.sh
lora_sft_14b.yaml   -> run_sft_lora_14b.sh
```

14B 配置使用专用长度过滤、ZeRO-3 offload 和梯度检查点。LoRA 合并使用 `merge_lora_3b.yaml` 或 `merge_lora_14b.yaml`；合并后的完整模型才能直接供 vLLM/Transformers 加载。

## 4. RL 训练

### 4.1 生成 RL 数据

```bash
cd /opt/data/private/GraphSSR-master/data_generation/pipelines
python preprocess.py
python rl_data_generator.py
```

`preprocess.py` 使用 PyG NeighborLoader/LinkNeighborLoader 采样 `nc`、`lp`、`lc` 子图，过滤过小/过大图并随机重排局部节点。`rl_data_generator.py` 使用 SFT 模型多次推理划分 easy/medium/hard，并持久化采样状态以支持断点续传。

### 4.2 Ray 和 vLLM

多节点运行前修改 `scripts/start_ray_cluster.sh` 的 `NODES`、Docker 镜像、挂载路径和容器配置：

```bash
cd /opt/data/private/GraphSSR-master/scripts
bash start_ray_cluster.sh
```

模型服务部署使用 `scripts/deploy_api.sh`，需保证 `SERVED_MODEL_NAME` 与调用方 endpoint 配置一致。

### 4.3 Stage 1

Stage 1 学习 SSR 结构真实性、子图选择一致性和答案正确性。检查 `run_rl_stage1.sh` 中的 `SFT_MODEL_PATH`、`TRAIN_FILE`、`VAL_FILE`、`RL_OUTPUT_DIR`、`RL_ENV_BIN`、`trainer.n_gpus_per_node` 和 `trainer.nnodes`。

```bash
cd /opt/data/private/GraphSSR-master/reinforcement_learning
CUDA_VISIBLE_DEVICES=0,1,2 bash run_rl_stage1.sh
# LoRA：
bash stage1_lora.sh
```

### 4.4 Stage 2

Stage 2 必须使用 Stage 1 checkpoint，并增加子图规模/去噪奖励。设置 `run_rl_stage2.sh` 中的 `actor_rollout_ref.model.path`、训练/验证 Parquet、`trainer.default_local_dir` 和 `trainer.second_stage_reward_lambda`：

```bash
cd /opt/data/private/GraphSSR-master/reinforcement_learning
CUDA_VISIBLE_DEVICES=0,1,2 bash run_rl_stage2.sh
# LoRA：
bash stage2_lora.sh
```

修改版 verl 增加 dataset+task 验证指标、全局准确率、最佳 checkpoint 保留/恢复，以及 FSDP offload 和 vLLM profiling 的同步显存修复。

## 5. 评测

训练完成后，通过 vLLM 部署模型并配置 `evaluation/evaluate_trained_model.py` 的 endpoint、模型名和数据路径：

```bash
cd /opt/data/private/GraphSSR-master/evaluation
bash run_evaluate.sh
```

`evaluate_trained_model_note.py` 可用于检查 prompt 重建、答案提取和指标统计。

## 6. 常见问题

- 路径错误：检查脚本和 YAML 中的绝对路径，不要只修改 README 命令。
- OOM：降低 per-device batch、prompt/response 长度，启用 ZeRO-3 offload 和 gradient checkpointing。
- SFT 数据过少：检查严格 SSR 格式、真实性、多样性和答案过滤结果。
- 恢复训练：保留 `global_step_*`、状态文件和最佳 checkpoint 元数据。

## Citation

```bibtex
@article{li2026beyond,
  title={Beyond One-Size-Fits-All: Adaptive Subgraph Denoising for Zero-Shot Graph Learning with Large Language Models},
  author={Li, Fengzhi and Zhang, Liang and Zuo, Yuan and Zhao, Ruiqing and Liu, YanSong and Ma, Yunfei and Meng, Fanyu and Feng, Junlan},
  journal={arXiv preprint arXiv:2603.02938},
  year={2026}
}
```
