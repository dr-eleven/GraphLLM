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

使用的数据位于：

```text
outputs/synthetic_data/sft_data/graph_sft_data.json
outputs/synthetic_data/rl_data/train_9799_graphr2_subgraph.parquet
outputs/synthetic_data/rl_data/test_600_graphr2_subgraph.parquet
```

训练模型：

```text
/opt/data/private/Graph-R1-main/locals/DeepSeek-R1-Distill-Qwen-14B
```
下载
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --local-dir /opt/data/private/GraphSSR-master/locals/DeepSeek-R1-Distill-Qwen-14B --exclude "*.DS_Store" "imgs/*"

## 3. SFT 训练

### 3.1 Full SFT

运行文件：

```text
supervised_finetuning/full_sft.yaml
```

确认 `model_name_or_path`、`dataset`、`dataset_dir`、`deepspeed`、`output_dir` 和 `cutoff_len`，然后运行：

```bash
cd /opt/data/private/GraphSSR-master/supervised_finetuning
bash run_sft_shell.sh
```

脚本可能会先按 tokenizer 过滤完整 prompt+response 长度，再通过所选环境的 Python 模块调用 `llamafactory` 进行训练。但是这里不要让他筛选了



## 4. RL 训练

### 4.1 Ray 和 vLLM
这里官方用的是这样的，我不知道你那里的配置，可能用不上。
多节点运行前修改 `scripts/start_ray_cluster.sh` 的 `NODES`、Docker 镜像、挂载路径和容器配置：

```bash
cd /opt/data/private/GraphSSR-master/scripts
bash start_ray_cluster.sh
```

模型服务部署使用 `scripts/deploy_api.sh`，需保证 `SERVED_MODEL_NAME` 与调用方 endpoint 配置一致。

### 4.2 Stage 1

Stage 1 学习 SSR 结构真实性、子图选择一致性和答案正确性。检查 `run_zhanghao_14b_stage1.sh` 中的 `SFT_MODEL_PATH`、`TRAIN_FILE`、`VAL_FILE`、`RL_OUTPUT_DIR`、`RL_ENV_BIN`、`trainer.n_gpus_per_node` 和 `trainer.nnodes`。

```bash
cd /opt/data/private/GraphSSR-master/reinforcement_learning
 bash run_zhanghao_14b_stage1.sh
```

### 4.3 Stage 2

Stage 2 必须使用 Stage 1 checkpoint，并增加子图规模/去噪奖励。设置 `run_zhanghao_14b_stage2.sh` 中的 `actor_rollout_ref.model.path`、训练/验证 Parquet、`trainer.default_local_dir` 和 `trainer.second_stage_reward_lambda`：

```bash
cd /opt/data/private/GraphSSR-master/reinforcement_learning
 bash run_zhanghao_14b_stage2.sh
```

修改版 verl 增加 dataset+task 验证指标、全局准确率、最佳 checkpoint 保留/恢复，以及 FSDP offload 和 vLLM profiling 的同步显存修复。

## 5. 评测

训练完成后，使用 `reinforcement_learning/run_rl_eval.sh` 进行验证测试，但是先让codex确认一下，这是我让它生成的，不确定效果，总之使用outputs/synthetic_data/rl_data/test_600_graphr2_subgraph.parquet作为测试集，模型使用rl_stage2训练模型，参数使用rl训练那一套应该就没问题。

```bash
cd /opt/data/private/GraphSSR-master/reinforcement_learning
bash run_rl_eval.sh
```

`evaluate_trained_model_note.py` 可用于检查 prompt 重建、答案提取和指标统计。

### 5.1 论文 Table 1 中 GraphSSR 的主要结果

下表摘录论文 Table 1 中与当前主要零样本评测设置对应的四组结果。`Candidate categories` 表示每条任务 prompt 中可供模型选择的候选类别数量，指标为 Accuracy（%）。

| 数据集 | 任务 | Candidate categories | GraphSSR Accuracy (%) |
|---|---|---:|---:|
| Cora | 节点分类 | 7 | 72.41 |
| WikiCS | 节点分类 | 10 | 79.40 |
| Products | 节点分类 | 47 | 68.49 |
| FB15K237 | 链接分类 | 10 | 78.94 |

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
