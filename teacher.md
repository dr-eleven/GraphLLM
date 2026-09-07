# GraphR1 与 GraphSSR：Teacher Model 生成 SFT `response` 的实现与运行指南

本文记录基于当前工作区代码、已下载数据，以及 GraphR1/GraphSSR 论文的核对结果。重点回答三个问题：teacher 收到什么 prompt、生成的文本怎样成为 SFT `response`、在当前机器上应如何启动生成流程。

> 审计日期：2026-08-24。项目目录分别为 `/opt/data/private/Graph-R1-main` 和 `/opt/data/private/GraphSSR-master`。

## 先说结论

| 项目 | Teacher | 数据生成代码是否公开在本仓库 | 最终的 prompt/response 数据 |
|---|---|---|---|
| GraphR1 | 论文使用 **DeepSeek-R1** 生成详细推理轨迹和答案；较长节点文本先由 **DeepSeek-V3** 摘要 | **没有**。仓库只提供生成后的 SFT JSONL 和训练代码 | `Graph-R1-main/data/graph_r1_sftdataset/sft_train_10k.json` |
| GraphSSR | 代码允许任意能提供 OpenAI-compatible Chat Completions API 的强 teacher；论文以 **DeepSeek-R1** 为例 | **有**。prompt 构造、并发 teacher 调用、质量过滤和最终导出均已公开 | `GraphSSR-master/outputs/synthetic_data/sft_data/graph_sft_data.json` |

两者关系如下：

```text
原始图数据
  └─ GraphR1：h-hop 子图线性化 → DeepSeek-R1 response → 过滤 → GraphR1 SFT 数据
       └─ GraphSSR：读取 GraphR1 prompt/response
            ├─ 从旧 response 提取 truth_label
            ├─ 把旧 prompt 改写为 Sample-Select-Reason prompt
            ├─ teacher 重新生成 SSR response
            ├─ 真实性 / 多样性 / 选择一致性 / 答案正确性过滤
            └─ GraphSSR SFT 数据
```

## 一、GraphR1 如何生成 SFT response

### 1. 论文描述的完整过程

GraphR1 论文 Section 2.2 给出的流程是：

1. 从 5 个领域的 11 个数据集采样节点、边和图级任务，包括节点分类、链接预测、图分类和链接分类。
2. 针对目标节点抽取 h-hop 子图，把节点文本特征和边关系线性化成自然语言；长标题/摘要等文本先用 DeepSeek-V3 自动摘要。
3. 将“节点文本 + 边关系 + 任务专用 prompt”组成查询 `Q_i`。
4. 把 `Q_i` 输入 DeepSeek-R1，让 teacher 一次生成显式推理轨迹 `R_i` 和最终预测 `Y_i`，即 `Q_i -> (R_i, Y_i)`。
5. 做三阶段质量控制：
   - 信息充分性：删除孤立节点和过于简单的子图；
   - 答案有效性：删除预测答案与 gold label 不一致、或包含敏感内容的样本；
   - 推理连贯性：只保留长度合理、逻辑连贯的 rationale。
6. 论文称最终保留 10,000 条推理样本，然后用 `(prompt, response)` 做全参数 SFT。

论文原文和方法说明见：

- GraphR1 论文：[ACL Anthology 页面](https://aclanthology.org/2025.emnlp-main.1220/)
- GraphR1 论文 PDF：[Graph-Reasoning Data Curation, Section 2.2](https://aclanthology.org/2025.emnlp-main.1220.pdf)

### 2. response 的实际格式

本地文件 `data/graph_r1_sftdataset/sft_train_10k.json` 实际是 **JSONL**，不是一个 JSON 数组；每行字段为：

```json
{
  "prompt": "线性化的图、任务说明和候选标签",
  "response": "<think>...详细推理...</think>\nAnswer: ...\nBrief_reasoning: ...",
  "dataset": "arxiv",
  "task": "lp"
}
```

当前本地文件实测有 10,919 行：`nc=4995`、`lp=2860`、`lc=1292`、`gc=1772`。这与文件名中的 `10k` 以及论文所述最终 10,000 条并不完全相同，应以实际文件为准。

### 3. 对应文件

| 文件 | 作用 |
|---|---|
| `Graph-R1-main/data/graph_r1_sftdataset/sft_train_10k.json` | 已经生成好的 prompt/teacher response；是观察 teacher 输出格式的直接证据 |
| `Graph-R1-main/data/graph_r1_sftdataset/README.md` | 数据卡占位文件，没有生成说明 |
| `Graph-R1-main/code/README.md` | 只说明如何拿现成数据执行 SFT/GRPO，不包含 teacher generation |
| `Graph-R1-main/code/LLaMA-Factory/data/dataset_info.json` | 把 `graph` 数据集映射到 `graph_10000_balance.json`，用于训练而不是生成 |
| `Graph-R1-main/code/LLaMA-Factory/examples/train_full/deepseek1.5B_full_sft.yaml` | SFT 配置，不调用 teacher |

重要结论：在 GraphR1 仓库内搜索不到 DeepSeek-R1/DeepSeek-V3 的批量调用、原始图转 prompt、三阶段过滤脚本或 API 配置。因此仓库能复现“拿已有 response 训练”，不能一键复现“从原始图生成 response”。

### 4. 如果要重新运行 GraphR1 teacher

有两种目标，必须区分。

#### 目标 A：只对仓库现有 prompt 重新生成 response

这是当前最现实的做法：保留 JSONL 中的 `prompt/dataset/task`，忽略原 `response`，调用 DeepSeek-R1 或一个可接受的替代 teacher，再写出新的 JSONL。

操作步骤：

1. 准备可访问的 DeepSeek-R1 teacher。若追求论文级复现，应使用论文对应的完整 DeepSeek-R1，而不是本机的 `DeepSeek-R1-Distill-Qwen-14B`；后者是较小的蒸馏模型，只能算近似 teacher。
2. 用 vLLM 暴露 OpenAI-compatible API，例如：

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /path/to/DeepSeek-R1 \
     --tensor-parallel-size 8 \
     --port 31200 \
     --served-model-name teacher_model
   ```

   完整 DeepSeek-R1 的资源需求远超普通单机 8 卡；资源不足时应使用托管 API 或明确记录所用替代模型。模型上下文长度必须大于“GraphR1 prompt token 数 + 最大生成 token 数”。

3. 用 HTTP 请求验证服务：

   ```bash
   curl http://127.0.0.1:31200/v1/models

   curl http://127.0.0.1:31200/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{
       "model":"teacher_model",
       "messages":[{"role":"user","content":"YOUR_GRAPH_PROMPT"}],
       "temperature":0.5,
       "max_tokens":4096
     }'
   ```

4. 编写/运行一个批处理 driver：逐行读取 `sft_train_10k.json`，发送 `messages=[{"role":"user","content": row["prompt"]}]`，把 `choices[0].message.content` 写回 `row["response"]`，并用 `index` 支持断点续跑。
5. 从原数据 response 的 `Answer:` 提取 gold label；从新 response 中同样提取答案。只保留两者一致且格式完整的样本。
6. 另外实现论文描述但仓库未公开细节的过滤：敏感内容过滤、最短/最长 reasoning 长度、空或截断 reasoning、逻辑连贯性审核。由于论文没有公开全部阈值，这一步只能近似复现，必须在实验记录里保存自定义规则。
7. 输出 JSONL，不要输出 JSON 数组；LlamaFactory 中将该文件登记为 `graph_10000_balance.json` 后即可做 SFT。

可以直接复用 GraphSSR 的 `utils/utils.py::reason()` 作为并发 API 调度器，但它只是 transport 层，**不会补上 GraphR1 缺失的 prompt 构造与三阶段过滤逻辑**。

#### 目标 B：从原始图完整重建 GraphR1 SFT 数据

还需要自行补齐以下未发布环节：11 个原始数据集的读取与采样、h-hop 子图抽取、DeepSeek-V3 节点文本摘要、Appendix B 中各任务 prompt 模板、gold label 对齐、DeepSeek-R1 批量推理以及三阶段过滤。因此不能仅依靠当前 GraphR1 仓库完成严格复现。若目的只是训练 GraphR1，建议直接使用已发布的 `sft_train_10k.json`；若研究数据蒸馏流程，则应把上述环节实现为独立的数据生成项目，并固定模型快照、采样参数、随机种子和过滤阈值。

## 二、GraphSSR 如何生成 SFT response

GraphSSR 的实现是完整且可追踪的。它不复用 GraphR1 的旧 response 作为新 response，而只从旧 response 提取正确标签，然后让 teacher 按新的 SSR prompt 重新生成整条轨迹。

### 1. 原始数据解析

入口文件：`data_generation/pipelines/graph_prepare.py`

- `get_graph_from_graphr1_sft()` 读取：
  `data/train_data/raw_data/graphr1/sft_train_data_filter2048_10000_balance.csv`。
- `extract_info_from_sft_data()` 从旧 GraphR1 `prompt` 中正则提取目标节点、节点文本、边和候选标签。
- 它从旧 GraphR1 `response` 的 `</think>` 之后解析 `Answer:`，作为 `truth_label`。
- 只接受 `nc/lp/lc`，跳过 `gc`。当前 10,919 条输入中理论上有 9,147 条进入 GraphSSR prompt 构造。
- 首次解析结果缓存到：
  `data/train_data/raw_data/graphr1/graphr1_sft_graph_info.jsonl`。

关键代码位置：

- `graph_prepare.py:65-108`：解析 GraphR1 prompt/response；
- `graph_prepare.py:198-236`：读取 CSV、跳过不支持任务、生成图信息缓存。

### 2. SSR prompt 构造

入口文件：`data_generation/pipelines/prompt_templete.py`

`construct_prompt_chosen_reasoning()` 将以下内容拼成一个 user message：

1. Graph analysis expert 的任务描述；
2. 完整子图：中心节点、节点文本、连接关系；
3. **Sample**：从完整子图生成候选子图，要求真实、结构/语义尽可能多样，并包含仅中心节点的候选；
4. **Select**：从候选组选择最适合任务的一个子图；
5. **Reason**：只能基于被选子图推理；
6. 候选标签；
7. 强制 XML-like 输出格式，包括每个 `<Subgraph_i>` 和 `<Chosen_subgraph_reasoning>`。

当前工作副本的 `group_size=5`，并将 `<Subgraph_0>` 到 `<Subgraph_4>` 全部显式展开；这是本地增强后的严格格式。关键位置为 `prompt_templete.py:17-135` 和 `prompt_templete.py:138-156`。

`data_generation/pipelines/sft_data_generator.py::construct_prompt_for_sft()` 为每条图记录调用该模板，输出：

```text
outputs/synthetic_data/sft_data/sft_prompt.jsonl
```

每行含 `sample_index/dataset/task/prompt/truth_label`。

### 3. Teacher API 调用如何把结果写入 response

入口：`data_generation/pipelines/sft_data_generator.py::sft_data_reasoning()`。

1. 读取 `sft_prompt.jsonl`，增加内部 `index`。
2. 检查已有 `sft_reasoning.jsonl` 中的 index，跳过已完成样本，因而支持断点续跑。
3. 采样参数固定为：

   ```python
   {"temperature": 0.5, "max_tokens": 16384}
   ```

4. 调用 `utils/utils.py::reason()`，由线程池把任务分配到一个或多个 API endpoint。
5. `worker()` 构造的请求消息只有一条 user message：

   ```python
   messages = [{"role": "user", "content": sample["prompt"]}]
   ```

6. `call_deepseek()` 向 `/v1/chat/completions` POST OpenAI-compatible payload，读取：

   ```python
   response_data["choices"][0]["message"]["content"]
   ```

7. 生成文本存到 `sample["response"]`，同时记录 `completion_tokens` 与 `latency`，逐条追加到：

   ```text
   outputs/synthetic_data/sft_data/sft_reasoning.jsonl
   ```

8. 请求失败时，Tenacity 在 HTTP 层最多尝试 3 次；worker 失败后等待 30 秒，上层调度器还会继续重提失败 index。完成后按 index 排序。

关键代码位置：

- `sft_data_generator.py:73-112`：断点、采样参数和 teacher 调用；
- `utils/utils.py:26-35`：teacher endpoint 配置；
- `utils/utils.py:41-65`：HTTP payload 和 response 提取；
- `utils/utils.py:71-89`：写入 `sample["response"]`；
- `utils/utils.py:93-234`：多 endpoint 并发调度、重试和 JSONL 追加；
- `utils/utils.py:238-246`：按 index 排序。

### 4. Teacher response 如何被过滤

GraphSSR 不会把所有 teacher 输出直接用于 SFT。

1. `data_generation/quality_check/quality_evaluator.py` 解析 teacher 的 XML-like response。
2. 当前工作副本检查：候选子图数量与编号、候选/选择块格式、节点和边真实性、是否包含 center-only 子图、规则式结构多样性、被选子图是否来自候选组、最终答案是否等于 `truth_label`。
3. `data_generation/quality_check/graph_diversity.py` 将 5 个候选子图两两组合，共产生 `C(5,2)=10` 个 pairwise prompt。
4. diversity model 对每对子图输出 0～1 分数；代码使用 `temperature=0.0, max_tokens=4096`。
5. 每条样本的 pairwise score 求平均，当前过滤阈值为 `> 0.3`。
6. 过滤后的 `sft_reasoning.jsonl` 和对应 diversity score 文件会被原地重写，只留下合格数据。
7. `reason_check_filter_sft_data()` 会再次为被淘汰的 index 调 teacher，反复执行“生成 → 多样性评分 → 过滤”，直到本轮答案正确率达到 0.9。
8. `construct_sft_data()` 最后把 JSONL 合并为 JSON 数组：

   ```text
   outputs/synthetic_data/sft_data/graph_sft_data.json
   ```

该数组仍保留 `prompt` 和 `response`，LlamaFactory 的映射在 `supervised_finetuning/dataset_info.json`。

论文层面，SSR-SFT 的四类过滤是：Subgraph Authenticity、Structural Diversity、Selection Consistency 和 Answer Check，最终得到 8,000+ 高质量示例。论文说明见：[GraphSSR Section 2.3](https://arxiv.org/html/2603.02938v2#S2.SS3)。

## 三、在当前工作区实际运行 GraphSSR teacher generation

### 0. 先保护现有结果

当前目录已经有约 254 MB 的预生成 `graph_sft_data.json`，但没有 `sft_prompt.jsonl`、`sft_reasoning.jsonl` 和 diversity 中间文件。完整运行会创建/重写同目录文件。建议先备份：

```bash
cd /opt/data/private/GraphSSR-master
cp -a outputs/synthetic_data/sft_data \
  outputs/synthetic_data/sft_data.backup-20260824
```

### 1. 修好环境依赖

当前 `/opt/data/private/.conda/envs/graphssr` 已有 pandas、numpy、requests、tqdm、torch，但实测缺少 `tenacity`，导入 `utils/utils.py` 会失败。安装后验证：

```bash
/opt/data/private/.conda/envs/graphssr/bin/pip install tenacity

/opt/data/private/.conda/envs/graphssr/bin/python -c \
  "import pandas,numpy,requests,tqdm,tenacity,torch; print('imports ok')"
```

### 2. 部署两个 OpenAI-compatible 服务

需要两个逻辑角色：

- teacher model：生成完整 SSR reasoning response；served name 必须和配置中的 `model_name` 一致，默认 `teacher_model`；
- diversity model：比较候选子图；默认 served name 为 `diversity_model`。

可以使用两个不同模型，也可以让同一个强模型启动两个服务/端口。示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /path/to/teacher-checkpoint \
  --tensor-parallel-size 4 \
  --port 31200 \
  --served-model-name teacher_model

CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve /path/to/diversity-checkpoint \
  --tensor-parallel-size 4 \
  --port 31201 \
  --served-model-name diversity_model
```

仓库的 `scripts/deploy_api.sh` 也能部署，但运行前必须改 `NODES/IMAGE_NAME/TAR_PATH/VOLUMES/MODEL_PATH/TP_SIZE/PORT/SERVED_MODEL_NAME`。该脚本每个节点使用固定容器名 `verl`；若在同一节点同时启动 teacher 和 diversity，必须使用不同容器名和端口，否则第二次部署会删除第一次的容器。

另外，teacher 的 `max_tokens=16384`，所以 vLLM 的最大上下文必须至少容纳最长 SSR prompt 再加 16,384 个输出 token。显存不足时应降低并发 `pool_size`，不要先随意截断 response，因为下游严格依赖完整的 5 个子图块和选择块。

### 3. 修改 endpoint 配置

编辑 `utils/utils.py`：

```python
URL_LIST_DS = {
    "teacher_model_0": {
        "name": "teacher_model_0",
        "model_name": "teacher_model",
        "pool_size": 8,
        "url": "http://127.0.0.1:31200/v1/chat/completions",
        "status": "active"
    }
}
```

编辑 `data_generation/quality_check/graph_diversity.py`：

```python
URL_LIST_DV = {
    "diversity_model_0": {
        "name": "diversity_model_0",
        "model_name": "diversity_model",
        "pool_size": 8,
        "url": "http://127.0.0.1:31201/v1/chat/completions",
        "status": "active"
    }
}
```

`pool_size=100/500` 是原集群的高并发占位值，不适合未经压测的单机服务；建议从 4～8 开始。

### 4. 修正当前代码中的 endpoint 名称不一致

当前 `utils/utils.py` 只定义了 `teacher_model_0`，但 `sft_data_generator.py:86` 硬编码：

```python
choose_name = ["teacher_model_0", "teacher_model_1", "teacher_model_2", "teacher_model_3"]
```

直接运行会在 `teacher_model_1` 处抛 `KeyError`。若只有一个服务，将其改为：

```python
choose_name = ["teacher_model_0"]
```

更稳妥的通用写法是：

```python
choose_name = list(URL_LIST_DS.keys())
```

diversity 侧当前已经使用 `list(URL_LIST_DV.keys())`，不需要同步硬编码多个名称。

### 5. 先做连通性和单样本 smoke test

确认模型名：

```bash
curl http://127.0.0.1:31200/v1/models
curl http://127.0.0.1:31201/v1/models
```

先只生成 prompt，不请求 teacher：

```bash
cd /opt/data/private/GraphSSR-master/data_generation/pipelines
/opt/data/private/.conda/envs/graphssr/bin/python -c \
  "from sft_data_generator import construct_prompt_for_sft; construct_prompt_for_sft()"

wc -l ../../outputs/synthetic_data/sft_data/sft_prompt.jsonl
sed -n '1p' ../../outputs/synthetic_data/sft_data/sft_prompt.jsonl
```

从第一行取出 `prompt`，手动发一次 `/v1/chat/completions`。至少检查 response 中存在完整的 `<Subgraph_0>`～`<Subgraph_4>`、`<Chosen_subgraph_reasoning>`、`<Answer>`，再开始全量任务。

### 6. 执行全流水线

必须从 `data_generation/pipelines` 目录运行，因为代码中的 `root_dir="../.."` 依赖当前工作目录：

```bash
cd /opt/data/private/GraphSSR-master/data_generation/pipelines
/opt/data/private/.conda/envs/graphssr/bin/python sft_data_generator.py \
  2>&1 | tee ../../logs/teacher_sft_generation.log
```

脚本依次执行：

```text
construct_prompt_for_sft()
  → reason_check_filter_sft_data()
      → sft_data_reasoning()
      → check_sft_reasoning_diversity()
      → filter_sft_data()
      → 未到 0.9 则继续循环
  → construct_sft_data()
```

中断后可以用同一命令继续；teacher generation 会按 `index` 跳过已写入 `sft_reasoning.jsonl` 的样本，diversity 也会跳过已有 `sample_index`。不要手工往 JSONL 中写半行；若进程被强杀，恢复前先检查末行能否被 `json.loads` 解析。

### 7. 验证输出

```bash
cd /opt/data/private/GraphSSR-master

wc -l outputs/synthetic_data/sft_data/sft_prompt.jsonl
wc -l outputs/synthetic_data/sft_data/sft_reasoning.jsonl
wc -l outputs/synthetic_data/sft_data/sft_reasoning_diversity_score.jsonl

/opt/data/private/.conda/envs/graphssr/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/synthetic_data/sft_data/graph_sft_data.json")
data = json.loads(path.read_text())
assert data and all("prompt" in x and "response" in x for x in data)
assert all("<Chosen_subgraph_reasoning>" in x["response"] for x in data)
print("samples:", len(data))
print("first keys:", sorted(data[0]))
PY
```

最终用于 LlamaFactory 的是 `graph_sft_data.json`，不是 `sft_prompt.jsonl` 或未经筛选的中间 response。

## 四、容易误解或踩坑的地方

1. **GraphR1 没有 teacher generation 脚本。** `LLaMA-Factory` 和 `TinyZero` 分别负责 SFT 与 GRPO，不能生成原始 SFT response。
2. **GraphSSR 中函数名 `call_deepseek` 不等于只能调用 DeepSeek。** 实际协议是 OpenAI-compatible `/v1/chat/completions`；真正模型由 endpoint 的 `model_name` 和 vLLM `--served-model-name` 决定。
3. **GraphSSR 需要两个角色。** teacher 负责写 SFT response，diversity model 负责 pairwise score；后者的输出不是学生 SFT response，但会决定样本能否保留。
4. **当前代码不是简单的“生成一次”。** 过滤会原地缩减 reasoning 文件，再对被淘汰 index 重采样，直到正确率达到 90%，因此费用和时间可能远高于 9,147 次调用。
5. **每条 GraphSSR 样本还会产生 10 次 diversity 请求。** 5 个候选子图两两比较，所以总 API 请求量很大。
6. **GraphSSR 输入中的 truth_label 来自 GraphR1 的旧 response，而不是单独的 gold 列。** 如果替换原始 CSV，必须保持 `Answer:` 格式可解析，或者修改 `extract_info_from_sft_data()` 直接读取可靠 gold 字段。
7. **不要把本地 `locals/GraphSSR` 当 teacher。** 它是已经训练出的 GraphSSR 模型 checkpoint；用于重新蒸馏会变成自蒸馏，和论文的强 teacher 设定不同。
8. **当前容器看不到 NVIDIA driver。** 本次检查的 `nvidia-smi` 返回无法与驱动通信，因此这里没有实际启动 teacher；运行前必须先在有可用 GPU/驱动的节点上部署 vLLM。

## 五、最短可执行清单

如果目标是“现在重新生成 GraphSSR SFT response”，按以下顺序操作即可：

1. 备份现有 `outputs/synthetic_data/sft_data`。
2. 在 `graphssr` 环境安装 `tenacity`。
3. 在可用 GPU 节点启动 `teacher_model` 和 `diversity_model` 两个 OpenAI-compatible endpoint。
4. 修改 `utils/utils.py::URL_LIST_DS` 和 `graph_diversity.py::URL_LIST_DV`。
5. 把 `sft_data_generator.py` 中 hard-coded `choose_name` 改成实际 endpoint 名称或 `list(URL_LIST_DS.keys())`。
6. curl 检查 `/v1/models`，再用一个真实 SSR prompt 做 smoke test。
7. 从 `GraphSSR-master/data_generation/pipelines` 运行 `python sft_data_generator.py`。
8. 检查三份中间 JSONL，并验证最终 `graph_sft_data.json` 的样本数、字段、XML-like 格式和答案正确性。




• 结论是：GraphR1 和 GraphSSR 都不是把“正确答案明确告诉 teacher”，再让 teacher 润色成 response。它们更接近：

  图数据 + 任务要求 + 候选标签
                  ↓
            teacher 自主推理
                  ↓
  候选子图/推理过程 + 预测答案
                  ↓
  用隐藏的 gold label 和结构规则筛选
                  ↓
  合格的 response 成为 SFT 监督信号

  ## GraphR1

  GraphR1 的 prompt 包含：

  - 线性化后的 h-hop 子图
  - 节点文本
  - 节点之间的边
  - 下游任务要求
  - 候选答案或答案格式

  teacher 使用 DeepSeek-R1 生成：

  <think>
  结构分析、语义分析、推理过程……
  </think>
  Answer: ...
  Brief_reasoning: ...

  然后用 gold label 检查 Answer 是否正确，并过滤信息不足、答案错误、推理不连贯的样本。

  本地 GraphR1 SFT 数据只有：

  {
    "prompt": "...",
    "response": "...",
    "dataset": "arxiv",
    "task": "lp"
  }

  没有单独的 truth_label 字段，说明发布的是过滤后的最终训练数据：

  Graph-R1-main/data/graph_r1_sftdataset/sft_train_10k.json

  根据 GraphR1 论文描述，gold answer 用于生成后的质量过滤，而不是作为 prompt 中已标明的正确答案直接交给 teacher。

  ## GraphSSR

  GraphSSR 的证据更明确。

  生成 prompt 时，代码虽然把 truth_label 保存在样本记录中：

  GraphSSR-master/data_generation/pipelines/sft_data_generator.py:47

  line = {
      "prompt": prompt,
      "truth_label": truth_label,
  }

  但真正请求 teacher 时，只发送：

  GraphSSR-master/utils/utils.py:71

  messages = [
      {"role": "user", "content": sample["prompt"]}
  ]

  所以 truth_label 没有作为“正确答案”传给 teacher。

  teacher 看到的是：

  - 完整、可能带噪的原始子图
  - 中心节点
  - 节点文本
  - 边关系
  - 下游任务，例如节点分类或链接预测
  - 所有候选标签
  - Sample–Select–Reason 的操作要求
  - 强制 response 格式

  例如链接预测 prompt 会给出：

  Options:
  <Yes, they have citation relationships>,
  <No, they do not have citation relationships>

  正确答案确实出现在候选列表里，但没有被标记为正确答案。

  teacher 必须自己完成：

  1. 从完整子图采样多个候选子图。
  2. 选择一个最适合当前任务的子图。
  3. 只基于选中的子图推理。
  4. 从候选标签中预测答案。

  生成的 response 类似：

  <Subgraph_0>...</Subgraph_0>
  <Subgraph_1>...</Subgraph_1>
  ...
  <Subgraph_4>...</Subgraph_4>

  <Chosen_subgraph_reasoning>
    <Chosen_subgraph>2</Chosen_subgraph>
    <Chosen_subgraph_reason>...</Chosen_subgraph_reason>
    <Central_node_ID>...</Central_node_ID>
    <Neighboring_node_ID>...</Neighboring_node_ID>
    <Connection_relationship>...</Connection_relationship>
    <Answer>...</Answer>
    <Brief_reasoning>...</Brief_reasoning>
  </Chosen_subgraph_reasoning>

  生成结束后，才使用隐藏的 truth_label 检查 teacher 的答案：

  GraphSSR-master/data_generation/quality_check/quality_evaluator.py:199

  同时检查：

  - teacher 是否虚构节点或边
  - 候选子图是否都来自原图
  - 是否包含多样化候选子图
  - 被选择的子图是否确实属于候选组
  - 最终分类结果是否正确

  因此，GraphSSR 属于“teacher 自主生成 + rejection sampling”，而不是“把答案告诉 teacher”。

  ## 去噪任务是否需要提供 clean graph

  对于 GraphSSR 这种“面向下游任务的子图去噪”，通常不应该把 clean graph 直接放进学生最终使用的 prompt。

  GraphSSR 使用的是：

  带噪/完整 graph
  + 下游任务
  + 候选标签
  → 找出任务相关子图
  → 基于该子图预测

  它没有要求训练数据中存在唯一的 clean graph。

  原因是这里的“噪声”并不一定是假边或错误节点，而可能只是：

  - 对当前分类任务不相关
  - 对另一个任务可能有用
  - 拓扑上真实，但语义上会干扰预测

  因此，任务相关的 clean graph 通常不是唯一的。例如，同一个论文节点：

  - 做研究领域分类时，方法相关邻居可能最重要；
  - 做链接预测时，共同邻居可能最重要；
  - 做作者关系预测时，作者和机构节点可能最重要。

  GraphSSR 的“clean graph”实际上是 teacher 针对当前任务选择的子图，而不是数据集事先提供的唯一标准答案。

  ## 如果你确实有 clean graph

  建议把 clean graph 当作“特权监督”或 verifier，而不是直接放进学生可见的 prompt。

  推荐流程：

  学生/teacher 可见输入：
  noisy graph + task + candidate labels

  隐藏监督：
  clean graph 或 gold relevant subgraph
  gold downstream label

  teacher 输出：
  预测的 clean/relevant subgraph
  + 选择理由
  + 下游推理
  + 最终答案

  过滤器：
  预测子图是否接近 clean graph
  最终答案是否等于 gold label
  是否存在虚构节点和边

  这样训练和推理保持一致：推理阶段没有 clean graph，模型仍然知道如何从 noisy graph 中识别相关结构。

  如果把 clean graph 直接放进 teacher prompt，也可以用于生成数据，但需要注意：

  - teacher 很可能只是复制 clean graph，而没有真正学习去噪依据；
  - response 可能引用只有 clean graph 中才有的信息；
  - 学生推理时看不到 clean graph，出现训练—推理不一致；
  - 容易形成数据泄漏。

  如果一定要使用 clean graph，最好让它只出现在单独的 synthesis prompt 中，然后检查生成的 response 是否能仅由 noisy graph 推导出来。最终保存给 SFT 的 student prompt 中必须删除 clean graph。

  ## 两类“去噪”需要区分

   去噪目标                               是否需要 clean graph
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   GraphSSR 式任务相关子图选择            不一定需要；主要依赖下游 gold label、结构真实性和子图选择规则
  ─────────────────────────────────────  ───────────────────────────────────────────────────────────────
   恢复被随机增删边之前的原图             需要；原始 clean graph 是明确监督目标
  ─────────────────────────────────────  ───────────────────────────────────────────────────────────────
   识别虚假节点或恶意注入边               最好需要 clean graph、攻击标注或边级标签
  ─────────────────────────────────────  ───────────────────────────────────────────────────────────────
   只希望提升节点分类准确率               不必提供 clean graph，可用分类 gold label 做 rejection/reward
  ─────────────────────────────────────  ───────────────────────────────────────────────────────────────
   希望模型输出精确的 keep/delete mask    最好提供 clean subgraph 或节点/边相关性标签

  ## 对你的任务的建议

  如果目标是“从带噪图中选择适合分类的子图”，推荐数据格式为：

  ### Teacher 输入

  原始/带噪 graph
  + 中心节点
  + 节点文本和边
  + 分类任务说明
  + 所有候选类别
  + 要求先选相关子图再分类

  不要告诉 teacher 哪个类别是正确的。

  ### 隐藏监督

  gold classification label
  可选：clean graph / relevant nodes / relevant edges

  ### Teacher response

  候选子图
  + 选择的子图
  + 保留/删除这些节点和边的理由
  + 分类推理
  + 最终分类结果

  ### 过滤条件

  子图必须是输入图的真实子集
  不能增加不存在的节点和边
  预测标签必须等于 gold label
  选择的子图必须足以支持答案
  若有 clean graph，再计算节点/边 precision、recall、F1

  对于困难样本，可以让 teacher 对同一个 prompt 生成多条 response，再保留“答案正确且子图质量最高”的一条。这比直接把 gold answer 告诉 teacher 更接近 GraphSSR，也更能避免事后编造理由。