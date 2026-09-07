# GraphSSR 复现指南（基于当前工作区审计）

审计时间：2026-06-23  
项目根目录：`/opt/data/private/GraphSSR-master`

本文不是复述 `README.md`，而是基于当前这份工作区的真实状态，整理一份可执行的复现说明，重点回答四件事：

1. 现在仓库里已经有什么。
2. 现在还缺什么。
3. 每一步应该把数据、模型、镜像放到哪里。
4. 代码里有哪些必须先处理的不一致项。

## 1. 先给结论

当前工作区里，GraphSSR 的主要训练数据和预生成产物大多已经在本地：

- 已有 GraphR1 SFT 原始 CSV：
  `data/train_data/raw_data/graphr1/sft_train_data_filter2048_10000_balance.csv`
- 已有 9 个训练图数据集的摘要和编码图：
  `data/train_data/raw_data/{arxiv,children,computer,history,photo,pubmed,reddit,sports,wn18rr}/`
- 已有 RL 验证源数据：
  `data/train_data/rl_data/test_1050.parquet`
- 已有 GOFA 两个评测 CSV：
  `data/test_data/raw_data/gofa_test_data/gofa_test_data_53114.csv`
  `data/test_data/raw_data/gofa_test_data/gofa_supply_data_35603.csv`
- 已有预生成 SFT 数据：
  `outputs/synthetic_data/sft_data/graph_sft_data.json`
- 已有预生成 RL 数据：
  `outputs/synthetic_data/rl_data/train_9799_graphr2_subgraph.parquet`
  `outputs/synthetic_data/rl_data/test_600_graphr2_subgraph.parquet`

当前真正缺的，主要不是这些数据本体，而是下面这些东西：

- 缺 teacher model、本地 diversity model、本地 SFT 基座模型、本地 final model checkpoint。
- 缺 RL 子图缓存目录：
  `data/train_data/processed_data/{dataset}/...`
- 缺评测阶段会用到的 3 个测试图文件：
  `data/test_data/raw_data/products/products_graph_bert_encoded.pt`
  `data/test_data/raw_data/wikics/wikics_graph_bert_encoded.pt`
  `data/test_data/raw_data/fb15k237/fb15k237_graph_bert_encoded.pt`
- 缺可直接运行部署/训练脚本的 GPU + Docker + 驱动环境。
- 缺少若干本地化配置：节点名、镜像 tar 路径、挂载路径、模型路径、vLLM 地址、Ray 集群路径。

如果你的目标只是“跑通评测”，不需要从头训练，最短路径是：

1. 下载作者发布的最终模型到本地。
2. 补齐上面 3 个测试图文件。
3. 部署 final model 的 vLLM 服务。
4. 修改 `evaluation/evaluate_trained_model.py` 里的 `URL_LIST_EVAL`。
5. 运行 `evaluation/run_evaluate.sh`。

如果你的目标是“从头复现整条训练链路”，则还需要自备 teacher/diversity/base model、SFT 镜像、verl 镜像，以及 GPU 集群。

## 2. 当前工作区审计结果

### 2.1 已存在的关键文件

| 类别 | 路径 | 当前状态 |
| --- | --- | --- |
| GraphR1 SFT 原始数据 | `data/train_data/raw_data/graphr1/sft_train_data_filter2048_10000_balance.csv` | 已存在 |
| 训练图摘要与编码图 | `data/train_data/raw_data/{dataset}/{dataset}_summary.jsonl`、`{dataset}_graph_bert_encoded.pt` | 已存在 |
| RL 验证源数据 | `data/train_data/rl_data/test_1050.parquet` | 已存在 |
| GOFA 主测试集 | `data/test_data/raw_data/gofa_test_data/gofa_test_data_53114.csv` | 已存在 |
| GOFA 补充测试集 | `data/test_data/raw_data/gofa_test_data/gofa_supply_data_35603.csv` | 已存在 |
| 预生成 SFT 训练数据 | `outputs/synthetic_data/sft_data/graph_sft_data.json` | 已存在 |
| 预生成 RL 训练数据 | `outputs/synthetic_data/rl_data/train_9799_graphr2_subgraph.parquet` | 已存在 |
| 预生成 RL 验证数据 | `outputs/synthetic_data/rl_data/test_600_graphr2_subgraph.parquet` | 已存在 |

### 2.2 当前不存在，但代码会生成或依赖的文件

| 类别 | 目标路径 | 来源/生成方式 | 当前状态 |
| --- | --- | --- | --- |
| GraphR1 图信息缓存 | `data/train_data/raw_data/graphr1/graphr1_sft_graph_info.jsonl` | `data_generation/pipelines/graph_prepare.py` 自动生成 | 不存在 |
| RL 子图缓存 | `data/train_data/processed_data/{dataset}/{dataset}_{task}_subgraph_bert_encoded.pt` | `data_generation/pipelines/preprocess.py` 生成 | 不存在 |
| SFT 中间 prompt | `outputs/synthetic_data/sft_data/sft_prompt.jsonl` | `sft_data_generator.py` 生成 | 不存在 |
| SFT 中间推理结果 | `outputs/synthetic_data/sft_data/sft_reasoning.jsonl` | teacher model 推理生成 | 不存在 |
| SFT 多样性打分 | `outputs/synthetic_data/sft_data/sft_reasoning_diversity_score.jsonl` | diversity model 推理生成 | 不存在 |
| RL 中间筛选结果 | `outputs/synthetic_data/rl_data/rl_filter_prompt.jsonl` | `rl_data_generator.py` 生成 | 不存在 |
| RL 采样进度缓存 | `outputs/synthetic_data/rl_data/have_sampled_index_dic.pkl` | `rl_data_generator.py` 生成 | 不存在 |
| RL 难度缓存 | `outputs/synthetic_data/rl_data/checked_sample_dic.pkl` | `rl_data_generator.py` 生成 | 不存在 |
| 本地模型输出目录 | `outputs/models/...` | SFT / RL / 直接下载 final model | 不存在 |
| 评测输出目录 | `outputs/test_result/...` | `evaluation/run_evaluate.sh` 生成 | 不存在 |

### 2.3 当前环境检查结果

检查环境：`/opt/conda/envs/GraphLLM/bin/python`，Python `3.10.0`

已安装并可导入：

- `torch 2.6.0+cu124`
- `torch_geometric`
- `pandas`
- `pyarrow`
- `requests`
- `datasets`
- `transformers`
- `ray 2.55.1`
- `vllm 0.8.5.post1`
- `fastapi`

缺失或不可直接用：

- 缺 `tenacity`
  `utils/utils.py` 会直接 `from tenacity import retry, stop_after_attempt`
- 当前节点没有 `docker` 命令
- 当前节点 `nvidia-smi` 报驱动不可用，不能直接在本节点本地拉起 vLLM 或训练
- 当前 Python 环境里没有 `llamafactory`、`deepspeed`
  这部分本来就更适合放在容器里跑

## 3. 代码里必须先知道的几个坑

这些问题不一定都要马上改代码，但如果不提前知道，后面会卡住。

### 3.1 `sft_data_generator.py` 默认会访问 4 个 teacher endpoint，但 `utils/utils.py` 只定义了 1 个

现状：

- `data_generation/pipelines/sft_data_generator.py` 里写的是：
  `choose_name = ["teacher_model_0", "teacher_model_1", "teacher_model_2", "teacher_model_3"]`
- `utils/utils.py` 里默认只给了 `teacher_model_0`

含义：

- 如果你不补上 `teacher_model_1/2/3`，或者不把 `choose_name` 改成只保留现有 endpoint，SFT 数据生成会直接报错。

### 3.2 `deploy_api.sh` 只能作为“单种角色的一次部署模板”使用

现状：

- 固定 `CONTAINER_NAME="verl"`
- 固定 `PORT=31200`
- 固定只起 1 种 `SERVED_MODEL_NAME`

含义：

- teacher、diversity、SFT、eval 四类服务如果部署在同一批节点上，不能直接重复执行同一个脚本而不改参数。
- 你必须至少区分：
  - `CONTAINER_NAME`
  - `PORT`
  - `MODEL_PATH`
  - `SERVED_MODEL_NAME`

### 3.3 评测除了 GOFA 的两个 CSV，还依赖 3 个 `.pt` 文件

`evaluation/evaluate_trained_model.py` 在计算答案集合时，会访问：

- `data/test_data/raw_data/products/products_graph_bert_encoded.pt`
- `data/test_data/raw_data/wikics/wikics_graph_bert_encoded.pt`
- `data/test_data/raw_data/fb15k237/fb15k237_graph_bert_encoded.pt`

原因：

- `gofa_test_data_53114.csv` 实际包含 `products`、`wikics`、`fb15k237`
- `gofa_supply_data_35603.csv` 实际包含 `products_10ways`、`products_5ways`、`wikics_5ways`
- 这些集合里真正参与当前评测逻辑的 `nc/lc` 任务，需要上述 `.pt` 文件提供标签集合

当前这 3 个文件本地都不存在，所以只靠 GOFA 的 CSV 还不能完整评测。

### 3.4 RL 数据文件名存在“文档/代码/现成产物”不一致

当前现成产物文件名是：

- `outputs/synthetic_data/rl_data/train_9799_graphr2_subgraph.parquet`
- `outputs/synthetic_data/rl_data/test_600_graphr2_subgraph.parquet`

但当前 `data_generation/pipelines/rl_data_generator.py` 重新生成时，按代码会输出：

- `train_9799_graph_rl_data.parquet`
- `test_600_graph_rl_data.parquet`

含义：

- 如果你直接复用仓库现成 parquet，`run_rl_stage1.sh` 里的旧命名能对上。
- 如果你从头重生成 parquet，需要把 RL 训练脚本中的文件路径改成当前代码真实生成的名字。

### 3.5 README 写的是 `verl v0.6.x`，但仓库内 vendored 版本文件是 `0.7.0.dev`

路径：

- `reinforcement_learning/verl/version/version`

含义：

- 不要简单假设“任意最新 `verlai/verl` 镜像”都和这份仓库中的改造代码严格兼容。
- 更稳妥的做法是：用官方镜像提供运行时依赖，但把当前仓库目录完整挂载进容器里，以仓库内代码为准。

## 4. 建议采用的本地目录布局

仓库本身没有统一规定模型放哪。为了减少后续改路径的成本，建议直接在项目里统一成下面这套布局：

```text
/opt/data/private/GraphSSR-master
├── .downloads/
│   ├── hf_dataset_snapshot/
│   └── docker_images/
├── data/
├── outputs/
│   ├── synthetic_data/
│   ├── models/
│   │   ├── teacher/
│   │   ├── diversity/
│   │   ├── base_sft_model/
│   │   ├── sft/
│   │   ├── rl_stage1/
│   │   ├── rl_stage2/
│   │   └── final_hf/GraphSSR/
│   └── test_result/
└── ...
```

后面文档里的“建议放到哪里”，都按这套路径写。

## 5. 下载清单：从哪里下载、下载到哪里、现在缺不缺

### 5.1 数据与预生成产物

统一数据源：

- 数据仓：[Hugging Face dataset `wfdlfz/GraphSSR`](https://huggingface.co/datasets/wfdlfz/GraphSSR/tree/main)

建议先整仓下载到一个暂存目录，再拷贝需要的文件：

```bash
cd /opt/data/private/GraphSSR-master
mkdir -p .downloads/hf_dataset_snapshot
huggingface-cli download wfdlfz/GraphSSR \
  --repo-type dataset \
  --local-dir /opt/data/private/GraphSSR-master/.downloads/hf_dataset_snapshot
```

然后按下面的目标路径放置：

| 资源 | 目标路径 | 当前状态 | 说明 |
| --- | --- | --- | --- |
| GraphR1 SFT CSV | `data/train_data/raw_data/graphr1/sft_train_data_filter2048_10000_balance.csv` | 已有 | 不需要再下 |
| 9 个训练图数据集摘要 | `data/train_data/raw_data/{dataset}/{dataset}_summary.jsonl` | 已有 | 不需要再下 |
| 9 个训练图编码数据 | `data/train_data/raw_data/{dataset}/{dataset}_graph_bert_encoded.pt` | 已有 | 不需要再下 |
| RL 验证源数据 | `data/train_data/rl_data/test_1050.parquet` | 已有 | 不需要再下 |
| GOFA 主测试 CSV | `data/test_data/raw_data/gofa_test_data/gofa_test_data_53114.csv` | 已有 | 不需要再下 |
| GOFA 补充测试 CSV | `data/test_data/raw_data/gofa_test_data/gofa_supply_data_35603.csv` | 已有 | 不需要再下 |
| 评测用 `products` 图文件 | `data/test_data/raw_data/products/products_graph_bert_encoded.pt` | 缺失 | 评测前必须补齐 |
| 评测用 `wikics` 图文件 | `data/test_data/raw_data/wikics/wikics_graph_bert_encoded.pt` | 缺失 | 评测前必须补齐 |
| 评测用 `fb15k237` 图文件 | `data/test_data/raw_data/fb15k237/fb15k237_graph_bert_encoded.pt` | 缺失 | 评测前必须补齐 |
| 预生成 SFT 数据 | `outputs/synthetic_data/sft_data/graph_sft_data.json` | 已有 | 可跳过 SFT 数据生成 |
| 预生成 RL 训练数据 | `outputs/synthetic_data/rl_data/train_9799_graphr2_subgraph.parquet` | 已有 | 可跳过 RL 数据生成 |
| 预生成 RL 验证数据 | `outputs/synthetic_data/rl_data/test_600_graphr2_subgraph.parquet` | 已有 | 可跳过 RL 数据生成 |

### 5.2 最终模型

统一模型源：

- 最终模型仓：[Hugging Face model `wfdlfz/GraphSSR`](https://huggingface.co/wfdlfz/GraphSSR/tree/main)

建议下载到：

- `outputs/models/final_hf/GraphSSR/`

下载命令：

```bash
cd /opt/data/private/GraphSSR-master
mkdir -p outputs/models/final_hf/GraphSSR
huggingface-cli download wfdlfz/GraphSSR \
  --local-dir /opt/data/private/GraphSSR-master/outputs/models/final_hf/GraphSSR
```

当前状态：

- 本地缺失。

### 5.3 teacher model、diversity model、SFT 基座模型

这一项必须单独说明：

- 这份仓库和当前 README 没有唯一锁定 teacher model 的 Hugging Face repo id。
- 也没有唯一锁定 diversity model 的 repo id。
- 也没有唯一锁定 SFT 基座模型的 repo id。

仓库里能明确推出的只有“模型角色”和“接口约束”，不能唯一推出“必须下载哪个 checkpoint”。

能从代码里明确读出的约束如下：

- `supervised_finetuning/full_sft.yaml` 使用 `template: deepseekr1`
- 多处解析逻辑依赖 `</think>` 和 XML 风格标签
- teacher/SFT/final eval endpoint 都走 OpenAI-compatible chat completion 接口

因此，比较稳妥的选择标准是：

1. 选择一个能稳定输出 `</think>` 风格推理内容的 reasoning 模型。
2. 保证它能在 vLLM 下以 OpenAI-compatible 方式对外提供 `/v1/chat/completions`。
3. 如果你要完全照当前 SFT 模板走，尽量选与 `deepseekr1` 模板兼容的模型家族。

建议本地目录：

- teacher model：`outputs/models/teacher/<your_teacher_model>/`
- diversity model：`outputs/models/diversity/<your_diversity_model>/`
- SFT 基座模型：`outputs/models/base_sft_model/<your_base_model>/`

当前状态：

- 都缺失。

## 6. 最短复现路径：只复现最终评测

如果你不打算从头训练，这是最省事的方案。

### 第 1 步：下载最终模型

下载到：

- `outputs/models/final_hf/GraphSSR/`

命令见上文 5.2。

当前缺口：

- 本地没有 final model。

### 第 2 步：补齐评测所需的 3 个 `.pt` 文件

需要补齐：

- `data/test_data/raw_data/products/products_graph_bert_encoded.pt`
- `data/test_data/raw_data/wikics/wikics_graph_bert_encoded.pt`
- `data/test_data/raw_data/fb15k237/fb15k237_graph_bert_encoded.pt`

来源：

- 上文 5.1 的 Hugging Face 数据仓。

当前缺口：

- 本地这 3 个文件都不存在。

### 第 3 步：部署 final model 的 vLLM 服务

需要修改：

- `scripts/deploy_api.sh`

至少要改：

- `NODES`
- `IMAGE_NAME`
- `TAR_PATH`
- `VOLUMES`
- `MODEL_PATH=/opt/data/private/GraphSSR-master/outputs/models/final_hf/GraphSSR`
- `PORT`
- `SERVED_MODEL_NAME=eval`

注意：

- `evaluate_trained_model.py` 里的 `URL_LIST_EVAL[*]["model_name"]` 默认就是 `eval`
- 所以 `vllm serve --served-model-name` 最好也设成 `eval`

当前缺口：

- 当前节点没有 Docker，也没有可用 GPU 驱动；这一步需要在你自己的 GPU 节点上做。

### 第 4 步：把评测 endpoint 写回代码

修改：

- `evaluation/evaluate_trained_model.py`

更新：

- `URL_LIST_EVAL`

示例原则：

- `url` 写成真实服务地址，例如 `http://<node-ip>:<port>/v1/chat/completions`
- `model_name` 与 `vllm serve --served-model-name` 保持一致，默认建议 `eval`

### 第 5 步：运行评测

```bash
cd /opt/data/private/GraphSSR-master/evaluation
bash run_evaluate.sh
```

输出会写到：

- `outputs/test_result/gofa_test_data_53114/...`
- `outputs/test_result/gofa_supply_data_35603/...`

## 7. 全量复现路径：从数据生成到 SFT 再到 RL

## 7.1 环境准备

### 控制端 Python 环境

至少先补一个当前缺失的依赖：

```bash
/opt/conda/envs/GraphLLM/bin/pip install tenacity
```

否则这些脚本会直接 import 失败：

- `utils/utils.py`
- `evaluation/evaluate_trained_model.py`
- `data_generation/pipelines/sft_data_generator.py`
- `data_generation/pipelines/rl_data_generator.py`

### 集群端基础要求

需要你自己的 GPU 节点具备：

- 可用 NVIDIA 驱动
- Docker
- 节点间 SSH
- 足够的 GPU 显存
- 可挂载项目目录到容器

当前工作区所在节点不满足这部分条件。

## 7.2 SFT 数据生成

### 可以直接跳过吗

可以。当前已经有：

- `outputs/synthetic_data/sft_data/graph_sft_data.json`

这意味着如果你的目标只是训练 SFT，不一定要重跑 `sft_data_generator.py`。

### 如果你要从头生成

#### 数据是否已就绪

已就绪：

- `data/train_data/raw_data/graphr1/sft_train_data_filter2048_10000_balance.csv`

当前计数：

- 原始 CSV 行数：`10919`
- 现成 `graph_sft_data.json` 样本数：`8265`

#### 还缺什么

- teacher model checkpoint
- diversity model checkpoint
- 能跑 vLLM 的 GPU 节点
- `URL_LIST_DS` / `URL_LIST_DV` 真实地址
- `teacher_model_1/2/3` 或者修改 `choose_name`

#### 建议放置路径

- teacher model：`outputs/models/teacher/<your_teacher_model>/`
- diversity model：`outputs/models/diversity/<your_diversity_model>/`

#### teacher / diversity 服务命名建议

| 角色 | `SERVED_MODEL_NAME` 建议值 | 对应代码位置 |
| --- | --- | --- |
| teacher | `teacher_model` | `utils/utils.py` 的 `URL_LIST_DS[*]["model_name"]` |
| diversity | `diversity_model` | `graph_diversity.py` 的 `URL_LIST_DV[*]["model_name"]` |

#### 需要修改的文件

- `scripts/deploy_api.sh`
- `utils/utils.py`
- `data_generation/quality_check/graph_diversity.py`
- 视情况修改 `data_generation/pipelines/sft_data_generator.py` 中的 `choose_name`

#### 运行命令

```bash
cd /opt/data/private/GraphSSR-master/data_generation/pipelines
python sft_data_generator.py
```

## 7.3 SFT 训练

### 还缺什么

- SFT 基座模型 checkpoint
- LlamaFactory Docker 镜像
- 远程节点名、挂载路径、镜像 tar 路径

### 官方推荐来源

- LlamaFactory：`README.md` 推荐官方 Docker Hub 镜像  
  <https://hub.docker.com/r/hiyouga/llamafactory>

### 建议本地放置

- 基座模型：`outputs/models/base_sft_model/<your_base_model>/`
- SFT 输出目录：`outputs/models/sft/`
- 镜像 tar：`.downloads/docker_images/llamafactory.tar`

### 如果你要把 Docker 镜像做成 tar

示例：

```bash
docker pull hiyouga/llamafactory
docker save -o /opt/data/private/GraphSSR-master/.downloads/docker_images/llamafactory.tar hiyouga/llamafactory
```

注意：

- 仓库没有锁定 tag，实际复现时建议你自己固定一个 tag 并记录下来。

### 需要修改的文件

- `supervised_finetuning/full_sft.yaml`
  - `model_name_or_path`
  - `output_dir`
- `supervised_finetuning/run_sft_shell.sh`
  - `HOST_NAMES`
  - `DATA_DIR`
  - `EXEC_DIR`
- `supervised_finetuning/run_sft_shell_node.sh`
  - `IMAGE_NAME`
  - `IMAGE_TAR`
  - `-v /your/mount/source:/your/mount/target`

### 推荐填写方式

- `model_name_or_path=/opt/data/private/GraphSSR-master/outputs/models/base_sft_model/<your_base_model>`
- `output_dir=/opt/data/private/GraphSSR-master/outputs/models/sft`
- `DATA_DIR=/opt/data/private/GraphSSR-master/outputs/synthetic_data/sft_data`
- `EXEC_DIR=/opt/data/private/GraphSSR-master/supervised_finetuning`

### 运行命令

```bash
cd /opt/data/private/GraphSSR-master/supervised_finetuning
bash run_sft_shell.sh
```

## 7.4 RL 数据准备

### 可以直接跳过吗

可以。当前已经有：

- `outputs/synthetic_data/rl_data/train_9799_graphr2_subgraph.parquet`
- `outputs/synthetic_data/rl_data/test_600_graphr2_subgraph.parquet`

当前计数：

- `train_9799_graphr2_subgraph.parquet`：`9799` 行
- `test_600_graphr2_subgraph.parquet`：`600` 行
- `data/train_data/rl_data/test_1050.parquet`：`1050` 行

### 如果你要从头重生成

#### 数据是否已就绪

已就绪：

- `data/train_data/raw_data/{arxiv,children,computer,history,photo,pubmed,reddit,sports,wn18rr}/...`
- `data/train_data/rl_data/test_1050.parquet`

#### 还缺什么

- `data/train_data/processed_data/{dataset}/...` 子图缓存
- 已训练好的 SFT checkpoint 或等价的 SFT 推理服务
- `URL_LIST_RL` 真正可访问的 vLLM 地址

#### 第一步：先生成子图缓存

运行：

```bash
cd /opt/data/private/GraphSSR-master/data_generation/pipelines
python preprocess.py
```

它会产出：

- `data/train_data/processed_data/{dataset}/{dataset}_{task}_subgraph_bert_encoded.pt`

当前这整批文件本地不存在。

#### 第二步：部署 SFT 模型服务

建议：

- 模型目录：`outputs/models/sft/`
- `SERVED_MODEL_NAME=sft`

原因：

- `rl_data_generator.py` 的 `URL_LIST_RL[*]["model_name"]` 默认就是 `sft`

#### 第三步：更新 RL 数据生成脚本

修改：

- `data_generation/pipelines/rl_data_generator.py`
  - `URL_LIST_RL`

#### 第四步：运行 RL 数据生成

```bash
cd /opt/data/private/GraphSSR-master/data_generation/pipelines
python rl_data_generator.py
```

注意：

- 如果你从头重生成 parquet，文件名会是 `train_*_graph_rl_data.parquet` 和 `test_*_graph_rl_data.parquet`
- 后续 RL 训练脚本要按真实产出文件名填写，不能盲信旧示例路径

## 7.5 RL 训练

### 还缺什么

- 可用的 Ray 集群节点
- verl 运行时 Docker 镜像
- 已训练好的 SFT checkpoint
- 本地化后的 `run_rl_stage1.sh` / `run_rl_stage2.sh`

### 官方推荐来源

- `README.md` 推荐 `verlai/verl` Docker 镜像
- 当前仓库内带了一份修改过的 `reinforcement_learning/` 代码

### 建议本地放置

- verl 镜像 tar：`.downloads/docker_images/verl.tar`
- Stage 1 输出：`outputs/models/rl_stage1/`
- Stage 2 输出：`outputs/models/rl_stage2/`
- 或保留仓库原逻辑，统一放到：
  `outputs/models/verl_checkpoints/verl_grpo_graph_rl/...`

### 如果你要把 Docker 镜像做成 tar

示例：

```bash
docker pull verlai/verl
docker save -o /opt/data/private/GraphSSR-master/.downloads/docker_images/verl.tar verlai/verl
```

### 先启动 Ray 集群

修改：

- `scripts/start_ray_cluster.sh`
  - `NODES`
  - `IMAGE_TAG`
  - `IMAGE_PATH`
  - `DOCKER_RUN_CMD` 里的挂载路径

建议：

- `IMAGE_PATH=/opt/data/private/GraphSSR-master/.downloads/docker_images/verl.tar`
- 把整个项目根目录挂进容器，例如：
  `-v /opt/data/private/GraphSSR-master:/opt/data/private/GraphSSR-master`

运行：

```bash
cd /opt/data/private/GraphSSR-master/scripts
bash start_ray_cluster.sh
```

### Stage 1

必须修改：

- `reinforcement_learning/run_rl_stage1.sh`

这份脚本里现在所有核心路径都还是旧的 `/gpfsprd/...` 硬编码，至少要替换：

- `data.train_files`
- `data.val_files`
- `actor_rollout_ref.model.path`
- `trainer.default_local_dir`
- 日志输出路径
- `trainer.n_gpus_per_node`
- `trainer.nnodes`

推荐路径：

- `data.train_files=/opt/data/private/GraphSSR-master/outputs/synthetic_data/rl_data/train_9799_graphr2_subgraph.parquet`
  或你自己刚重生成的 `train_*_graph_rl_data.parquet`
- `data.val_files=/opt/data/private/GraphSSR-master/outputs/synthetic_data/rl_data/test_600_graphr2_subgraph.parquet`
  或你自己刚重生成的 `test_*_graph_rl_data.parquet`
- `actor_rollout_ref.model.path=/opt/data/private/GraphSSR-master/outputs/models/sft`
- `trainer.default_local_dir=/opt/data/private/GraphSSR-master/outputs/models/verl_checkpoints/verl_grpo_graph_rl/verl_grpo_graph_rl_stage1`

### Stage 2

必须修改：

- `reinforcement_learning/run_rl_stage2.sh`

同样要替换全部 `/gpfsprd/...` 路径，尤其是：

- `actor_rollout_ref.model.path`
  应指向 Stage 1 输出 checkpoint，例如：
  `.../verl_grpo_graph_rl_stage1/global_step_*/actor/huggingface`
- `data.train_files`
- `data.val_files`
- `trainer.default_local_dir`
- 日志输出路径
- `trainer.n_gpus_per_node`
- `trainer.nnodes`

### RL 运行命令

进入 head 节点后，在容器内执行：

```bash
cd /opt/data/private/GraphSSR-master/reinforcement_learning
bash run_rl_stage1.sh
bash run_rl_stage2.sh
```

## 7.6 最终评测

如果你是自己从头训练到 Stage 2，评测时可以直接部署你自己的最终 checkpoint。  
如果你只是想对齐作者结果，也可以直接用上文 5.2 下载的最终模型。

### 还缺什么

- final model 或你自己的最终 checkpoint
- 评测用 3 个 `.pt` 文件
- `URL_LIST_EVAL` 真实地址

### 服务命名建议

- `SERVED_MODEL_NAME=eval`

### 运行命令

```bash
cd /opt/data/private/GraphSSR-master/evaluation
bash run_evaluate.sh
```

## 8. 建议你优先采取的实际路线

如果你现在的目标是先验证“这份仓库能不能跑通”，建议按下面顺序来，不要一上来就从头全训：

1. 先补 `tenacity`。
2. 先下载 final model 到 `outputs/models/final_hf/GraphSSR/`。
3. 先从数据仓补齐评测用 3 个 `.pt` 文件。
4. 先只跑 evaluation，确认评测链路通。
5. 再决定是否重跑 SFT 数据生成。
6. 再决定是否重跑 SFT 训练。
7. 最后再做 RL 数据生成和两阶段 RL。

这样做的原因很简单：

- 当前仓库已经带了现成 `graph_sft_data.json`
- 当前仓库已经带了现成 RL train/test parquet
- 真正昂贵的部分是 teacher/diversity/SFT/RL 服务和多机训练
- 先跑评测，能最快判断你的部署环境和模型服务链路是否正确

## 9. 当前仍无法从仓库内唯一确定的内容

下面这些信息，我在当前仓库代码、README 和本地文件中都没有找到唯一答案，所以不能替你伪造一个“唯一正确”的下载名：

1. teacher model 的唯一 Hugging Face repo id
2. diversity model 的唯一 Hugging Face repo id
3. SFT 基座模型的唯一 Hugging Face repo id
4. LlamaFactory 镜像的唯一固定 tag
5. verl 镜像的唯一固定 tag

这意味着：

- 这份仓库给的是“训练流程”和“接口契约”
- 不是“所有外部模型/镜像都锁死版本的一键复现包”

如果你要做严格论文级复现，建议把你最终选择的：

- teacher model
- diversity model
- SFT base model
- LlamaFactory image tag
- verl image tag

单独记录成一份环境清单，否则后续结果很难严格对齐。

