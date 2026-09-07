# GraphSSR 项目阅读与运行指南

本文档基于对 `GraphSSR-master` 目录源码、README、关键脚本和实际命令验证的阅读整理而成。

## 1. 先说结论

### 1.1 这个项目是什么

`GraphSSR` 是一个完整流程型项目，覆盖四个阶段：

1. SFT 数据生成
2. SFT 训练
3. RL 数据生成与两阶段 RL 训练
4. 最终评测

它不是像 `Graph-R1` 那样由两个独立子项目拼出来，而是一个围绕 `GraphR1` 数据、LlamaFactory 和 veRL 改造版构建的完整流水线。

### 1.2 当前环境能不能直接运行

结论是：当前 `GraphLLM` 环境下不能直接完整跑通。

当前已经确认的阻塞点有四类：

1. 数据预处理缺依赖。
   - `data_generation/pipelines/preprocess.py` 需要 `torch_geometric`
   - 当前环境里没有这个包

2. SFT 数据生成缺原始数据。
   - `sft_data_generator.py` 在正确目录下可以继续执行
   - 但立刻因为缺少 `data/train_data/raw_data/graphr1/sft_train_data_filter2048_10000_balance.csv` 失败

3. RL 主入口缺 Hydra 配置组。
   - `python -m verl.trainer.main_ppo --cfg job` 报：
   - `Could not find 'data/legacy_data'`
   - 且仓库里确实不存在 `reinforcement_learning/verl/trainer/config/data/legacy_data.yaml`

4. 大量脚本仍然保留环境占位符。
   - `YOUR_NODE_*`
   - `/path/to/your/...`
   - `http://YOUR_NODE:PORT/...`
   - 这些都必须先替换

### 1.3 当前环境下已经验证到什么程度

已经确认：

- `evaluation/evaluate_trained_model.py --help` 可以正常显示参数帮助
- SFT 数据生成脚本能启动到读数据阶段
- 数据预处理脚本会因缺 `torch_geometric` 终止
- RL 主入口会在配置装载阶段终止

这说明：

- 项目的评测脚本至少能被解释器正确解析
- SFT / RL 流程不是“脚本名错了”，而是确实依赖大量外部资源和额外环境

## 2. 如果你想运行 GraphSSR，应该运行什么

下面按目标区分。

### 2.1 想部署教师模型、SFT 模型或评测模型的 API 服务

运行：

```bash
cd scripts
bash deploy_api.sh
```

主文件：

- `scripts/deploy_api.sh`

用途：

- 在多台机器上加载 Docker 镜像
- 启动容器
- 在容器里拉起 vLLM 服务

注意：

- 里面的节点名、镜像名、挂载目录、模型路径和端口全是占位符

### 2.2 想生成 SFT 训练数据

运行：

```bash
cd data_generation/pipelines
python sft_data_generator.py
```

主文件：

- `data_generation/pipelines/sft_data_generator.py`

这个脚本没有 `argparse`，一执行就会直接跑完整流程：

1. 构造 SFT prompt
2. 调教师模型做推理
3. 做多样性和正确性筛选
4. 生成最终 `graph_sft_data.json`

前提：

- 必须先准备 `GraphR1` SFT 原始 CSV
- 必须先在 `utils/utils.py` 和 `data_generation/quality_check/graph_diversity.py` 里配置好教师模型和多样性模型的服务地址

### 2.3 想生成 RL 训练前的子图数据

运行：

```bash
cd data_generation/pipelines
python preprocess.py
```

主文件：

- `data_generation/pipelines/preprocess.py`

用途：

- 用 `torch_geometric` 的 `NeighborLoader` / `LinkNeighborLoader` 对原图采样
- 把原图转成后续 RL 数据生成会用到的子图数据

前提：

- 需要 `torch_geometric`
- 需要各数据集的 `*_graph_bert_encoded.pt`

### 2.4 想生成 RL 训练 / 验证 parquet

运行：

```bash
cd data_generation/pipelines
python rl_data_generator.py
```

主文件：

- `data_generation/pipelines/rl_data_generator.py`

这个脚本同样没有 `argparse`，执行后会直接跑完整流程：

1. 采样图任务样本
2. 调 SFT 模型进行多次推理，给样本打 easy / medium / hard 难度
3. 生成训练 parquet
4. 把 `test_1050.parquet` 转成验证 parquet

前提：

- `preprocess.py` 已成功生成子图数据
- `data_generation/pipelines/rl_data_generator.py` 里的 `URL_LIST_RL` 已改成你自己的服务地址
- `data/train_data/rl_data/test_1050.parquet` 已存在

### 2.5 想做 SFT 训练

运行：

```bash
cd supervised_finetuning
bash run_sft_shell.sh
```

主文件：

- `supervised_finetuning/run_sft_shell.sh`
- `supervised_finetuning/run_sft_shell_node.sh`
- `supervised_finetuning/full_sft.yaml`
- `supervised_finetuning/dataset_info.json`

用途：

- 多机分发 Docker 容器
- 在每台机器里执行 `llamafactory-cli train examples/train_full/${YAML_FILE}`

前提：

- 节点名、项目路径、镜像名、挂载路径都已经替换
- `outputs/synthetic_data/sft_data/graph_sft_data.json` 已生成

### 2.6 想启动 RL 用的 Ray 集群

运行：

```bash
cd scripts
bash start_ray_cluster.sh
```

主文件：

- `scripts/start_ray_cluster.sh`

用途：

- 在多节点 Docker 容器中拉起 Ray head 和 Ray workers

前提：

- 节点名、Docker 镜像路径、挂载路径都已经替换

### 2.7 想做 RL 第一阶段训练

运行：

```bash
cd reinforcement_learning
bash run_rl_stage1.sh
```

主文件：

- `reinforcement_learning/run_rl_stage1.sh`

实际调用：

- `python3 -m verl.trainer.main_ppo ...`

用途：

- Stage 1: 只做真实性 / 一致性 / 正确性奖励

前提：

- Ray 集群已启动
- RL 训练 parquet 和验证 parquet 已生成
- SFT 模型路径已配置
- `verl` 训练配置能正确装载

### 2.8 想做 RL 第二阶段训练

运行：

```bash
cd reinforcement_learning
bash run_rl_stage2.sh
```

主文件：

- `reinforcement_learning/run_rl_stage2.sh`

用途：

- Stage 2: 在 Stage 1 基础上继续加入子图规模奖励

前提：

- Stage 1 checkpoint 已生成

### 2.9 想做最终评测

运行：

```bash
cd evaluation
bash run_evaluate.sh
```

主文件：

- `evaluation/run_evaluate.sh`
- `evaluation/evaluate_trained_model.py`

用途：

- 评测 GOFA 测试集和补充测试集

前提：

- 已部署评测模型 API
- `URL_LIST_EVAL` 已改成你自己的服务地址
- `data/test_data/raw_data/gofa_test_data/*.csv` 已存在

## 3. 目录和文件作用

### 3.1 顶层

- `README.md`
  - 顶层总文档
  - 把整个流程拆成 SFT 数据、SFT、RL 数据、RL、评测几个阶段

- `appendix/Online_Appendix.pdf`
  - 补充材料

### 3.2 `data_generation/`

这是数据生成阶段的核心目录。

- `pipelines/graph_prepare.py`
  - 负责从 GraphR1 SFT 数据、GraphR1 RL 数据或子图采样结果中抽取图结构信息
  - 也是很多 prompt 构造函数的底层数据准备模块

- `pipelines/prompt_templete.py`
  - 负责拼接 SSR 风格 prompt

- `pipelines/sft_data_generator.py`
  - 生成 SFT 训练数据

- `pipelines/preprocess.py`
  - 用 PyG 采样子图

- `pipelines/rl_data_generator.py`
  - 生成 RL 训练 / 验证 parquet

- `quality_check/quality_evaluator.py`
  - 检查子图真实性、选择一致性和答案正确性

- `quality_check/graph_diversity.py`
  - 给 SFT reasoning 结果打结构多样性分数

### 3.3 `supervised_finetuning/`

这是 SFT 阶段。

- `full_sft.yaml`
  - LlamaFactory 的训练配置

- `dataset_info.json`
  - 数据集信息配置

- `run_sft_shell.sh`
  - 多机调度总脚本

- `run_sft_shell_node.sh`
  - 单节点容器准备与训练触发脚本

### 3.4 `reinforcement_learning/`

这是 RL 阶段，里面嵌入了一份经过修改的 `verl`。

- `run_rl_stage1.sh`
  - 第一阶段 RL 训练

- `run_rl_stage2.sh`
  - 第二阶段 RL 训练

- `requirements.txt`
  - RL 子工程依赖
  - 但它并不覆盖顶层 `data_generation` 的所有依赖，例如 `torch_geometric`

- `pyproject.toml` / `setup.py`
  - `verl` 包的安装配置

- `verl/`
  - 被修改过的 RL 框架主体

- `verl/trainer/main_ppo.py`
  - RL 主入口

- `verl/trainer/config/`
  - Hydra 配置目录
  - 当前仓库里缺少 `data/legacy_data.yaml`

### 3.5 `scripts/`

- `deploy_api.sh`
  - 多节点部署 vLLM 服务

- `start_ray_cluster.sh`
  - 多节点拉起 Ray + Docker

### 3.6 `evaluation/`

- `evaluate_trained_model.py`
  - 构造 GOFA 评测 prompt
  - 调模型服务
  - 解析答案并统计准确率

- `run_evaluate.sh`
  - 批量评测两个 GOFA 文件

### 3.7 `utils/`

- `utils.py`
  - 对模型服务做并发 HTTP 调用
  - 被 SFT 数据生成和 RL 数据生成复用

## 4. 当前环境的实际检查结果

下面这些结论都来自当前环境里的真实命令验证。

### 4.1 评测脚本

执行：

```bash
cd /opt/data/private/GraphSSR-master
python evaluation/evaluate_trained_model.py --help
```

结果：

- 成功
- 能正常显示 `--data_path`、`--save_dir`、`--choose_server` 等参数

说明：

- 评测脚本本身可被解释器正确加载

### 4.2 SFT 数据生成脚本

执行：

```bash
cd /opt/data/private/GraphSSR-master/data_generation/pipelines
python sft_data_generator.py
```

结果：

- 能开始执行
- 但在读取原始 CSV 时失败
- 报错：`FileNotFoundError: ../../data/train_data/raw_data/graphr1/sft_train_data_filter2048_10000_balance.csv`

说明：

- 脚本运行目录确实应该是 `data_generation/pipelines`
- 当前主要缺的是原始数据，不是脚本名错了

### 4.3 子图预处理脚本

执行：

```bash
cd /opt/data/private/GraphSSR-master/data_generation/pipelines
python preprocess.py
```

结果：

- 失败
- 报错：`ModuleNotFoundError: No module named 'torch_geometric'`

说明：

- 当前环境缺 PyG 相关依赖

### 4.4 RL 主入口

执行：

```bash
cd /opt/data/private/GraphSSR-master/reinforcement_learning
python -m verl.trainer.main_ppo --cfg job
```

结果：

- 失败
- 报错：`Could not find 'data/legacy_data'`

进一步确认：

- 仓库里不存在 `reinforcement_learning/verl/trainer/config/data/`

说明：

- 当前 RL 子工程不是简单改路径就能启动
- 至少还缺一个 Hydra 配置组

## 5. 当前不能直接跑通的根本原因

`GraphSSR` 当前不能直接一键跑通，不是单一原因，而是下面几类前提同时缺失：

1. 缺外部原始数据和 parquet 数据
2. 缺 vLLM 服务和多节点容器环境
3. 缺 `torch_geometric` 这类预处理依赖
4. 缺 RL 配置文件 `data/legacy_data.yaml`
5. 多个脚本仍然保留占位节点名、占位路径和占位服务地址

## 6. 如果你要真正把它跑起来，建议的顺序

1. 先把所有 `YOUR_NODE_*`、`/path/to/your/...` 和 `http://YOUR_NODE:PORT/...` 替换成你自己的环境参数。
2. 先补齐原始数据：
   - GraphR1 SFT CSV
   - 多个图数据集的 `summary.jsonl` 和 `graph_bert_encoded.pt`
   - `test_1050.parquet`
   - GOFA 测试集 CSV
3. 补 `torch_geometric` 等数据预处理依赖。
4. 先确认 `deploy_api.sh` 能把教师模型或 SFT 模型服务拉起来。
5. 跑 `preprocess.py` 生成子图数据。
6. 跑 `sft_data_generator.py` 和 `rl_data_generator.py` 生成训练数据。
7. 跑 `supervised_finetuning/run_sft_shell.sh` 做 SFT。
8. 修复 `reinforcement_learning/verl/trainer/config/data/legacy_data.yaml` 缺失问题后，再启动 Ray 集群并运行 RL。
9. 最后再用 `evaluation/run_evaluate.sh` 做评测。

## 7. 最短回答

如果你只想知道“GraphSSR 我该运行什么文件”：

- 部署模型服务：`scripts/deploy_api.sh`
- 生成 SFT 数据：`data_generation/pipelines/sft_data_generator.py`
- 生成 RL 子图：`data_generation/pipelines/preprocess.py`
- 生成 RL parquet：`data_generation/pipelines/rl_data_generator.py`
- 做 SFT：`supervised_finetuning/run_sft_shell.sh`
- 启动 Ray：`scripts/start_ray_cluster.sh`
- RL 第一阶段：`reinforcement_learning/run_rl_stage1.sh`
- RL 第二阶段：`reinforcement_learning/run_rl_stage2.sh`
- 评测：`evaluation/run_evaluate.sh`

如果你只想知道“它现在能不能直接跑”：

- 不能
- 主要卡在缺数据、缺 `torch_geometric`、缺 RL Hydra 配置和大量占位环境参数
