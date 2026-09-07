# `outputs/evaluation` 评测汇总

本文整理 `GraphSSR/outputs/evaluation` 下已有的四组评测结果，并说明它们与 `outputs/test_result/result.md` 的对应关系。所有结果均为 GraphSSR 结构化输出评测，不是普通分类准确率评测。

## 1. 目录与共同评测口径

`outputs/evaluation` 下包含以下四个结果目录：

| 目录 | 模型/来源 | 样本 | 解码与长度设置 |
|---|---|---:|---|
| `ds3b_lora_sft_ssr_600` | Qwen2.5-3B-Instruct + `outputs/sft/ds3b_lora` 的合并模型；较早的 3B 基线 | 600 | greedy；response 8192；context 12288 |
| `ds3b_lora_sft_ssr_600_r10240_ctx14336` | 同一 `ds3b_lora` 3B SFT 模型；将输出/上下文上限提高后的对照 | 600 | greedy；response 10240；context 14336 |
| `ds3b_lora_sft_20260822_ssr_600_r10240_ctx14336` | 2026-08-22 完成训练的 3B LoRA SFT 新合并模型 `outputs/sft/ds3b_lora_merged_20260822` | 600 | greedy；response 10240；context 14336 |
| `qwen2.5_14b_lora_sft_20260823_ssr_600_r10240_ctx14336` | Qwen2.5-14B-Instruct + `outputs/sft/qwen2.5_14b_lora_merged` | 600 | greedy；response 10240；context 14336 |

四组结果都使用 600 条验证样本，每个任务 150 条，任务为 `fb15k237/lc`、`cora_node/nc`、`products/nc`、`wikics/nc`。结果由同一类 SFT SSR 评测脚本生成/审计，并同时记录严格指标和旧奖励函数指标。

需要区分两种比较：前两个 3B 目录主要用于**长度预算对照**，后两个目录是**新 3B SFT 模型与 14B SFT 模型**的结果。改变 response/context 上限会改变截断、EOS 和重复率，不能把这种差异直接归因于模型规模。

## 2. 全局指标对比

以下数值来自各目录的 `summary.json`；百分比均以 600 条为分母。

| 评测目录（简称） | 答案正确 | XML 完整 | 五子图均严格真实 | EOS | 长度截断 | 高重复 | 严格 `real_pred` | 旧版 `legacy_real_pred` | 旧奖励均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3B，8192 基线 | 12.50% | 19.00% | 5.33% | 19.00% | 81.00% | 77.67% | 3.33% | 5.00% | 0.0934 |
| 3B，10240 对照 | 13.83% | 20.67% | 6.00% | 20.67% | 79.33% | 78.33% | 3.67% | 5.33% | 0.0965 |
| 3B，2026-08-22 新 SFT | 13.50% | 19.83% | 5.00% | 19.67% | 80.33% | 79.67% | 3.33% | 5.50% | 0.0981 |
| 14B，2026-08-23 SFT | 53.17% | 73.50% | 30.67% | 73.67% | 26.33% | 26.33% | 22.00% | 24.83% | 0.2725 |

### 3B 的 8192 与 10240 对照

这两个目录使用同一个 `ds3b_lora` 合并模型，主要差别是生成上限从 8192 提高到 10240、上下文从 12288 提高到 14336。提高预算后：

- 答案正确数从 75 提高到 83，严格 `real_pred` 从 20 提高到 22；
- 截断数从 486 降到 476，但高重复数从 466 升到 470；
- 平均输出长度从 7719.80 增至 9357.03 token；
- 10240 上限没有解决长输出、重复和不停止问题，只带来小幅可解析性提升。

因此这两个结果不是两个不同训练阶段的模型，而是同一 3B SFT 模型的不同推理预算实验。

### 3B 旧模型与 2026-08-22 新 SFT 模型

`ds3b_lora_sft_20260822_ssr_600_r10240_ctx14336/REPORT.md` 将 2026-08-21 合并模型与 2026-08-22 新合并模型做了同预算比较：

- 答案正确：83 -> 81；
- XML 完整：124 -> 119；
- 五个严格真实子图：36 -> 30；
- 严格 `real_pred`：22 -> 20；
- 长度截断：476 -> 482；
- 高重复：470 -> 478。

也就是说，新 3B SFT 训练虽然完成并正确合并，但在该留出集上的整体 SSR 指标略差于前一版，训练 loss 下降不能直接等同于测试质量提升。

### 3B 与 14B

在同为 response 10240/context 14336 的设置下，14B 明显优于 3B：答案正确率 53.17% 对 13.50%，严格 `real_pred` 22.00% 对 3.33%。14B 的截断和高重复也显著较少。

但这仍不是严格的只改变参数规模的实验：两者训练数据过滤、训练配置和 checkpoint 不同；因此应表述为“当前 14B SFT 方案的评测结果明显优于当前 3B SFT 方案”，不能把全部增益归因于参数量。

## 3. 每个指标的含义

- `answer_correct`：`<Answer>` 被解析出来，且与 ground truth 完全相等（忽略大小写和首尾空格）。
- `xml_complete`：输出具有完整 GraphSSR XML，包含 `Subgraph_0` 至 `Subgraph_4` 和一个 chosen-subgraph reasoning 块。
- `strict_real_subgraphs`：五个子图都能在 prompt 提供的原图中找到，节点/边没有虚构；同时满足五个子图不同且至少有一个只含中心节点的子图。
- `stopped_on_eos` / `length_truncated`：生成分别因 EOS 正常结束，或达到最大 response token 上限而结束。
- `high_repetition`：归一化长行重复至少 8 次，或 32-token n-gram 重复至少 5 次。
- 严格 `real_pred`：`xml_complete`、严格真实子图、chosen 一致、答案正确、EOS、无高重复全部满足。
- `legacy_real_pred`：调用旧的 `reinforcement_learning/verl/utils/reward_score/subgraph_size.py` 得到的兼容指标；它的结构约束比严格指标宽松，因此通常高于严格 `real_pred`。

## 4. 与 `outputs/test_result/result.md` 的对应关系

`result.md` 中的“历史”行可以直接对应到 `outputs/evaluation`：

- **历史 3B**：`ds3b_lora_sft_20260822_ssr_600_r10240_ctx14336/summary.json`，即 2026-08-22 的 3B SFT 新合并模型，结果为答案正确 13.50%、严格 `real_pred` 3.33%。
- **历史 14B**：`qwen2.5_14b_lora_sft_20260823_ssr_600_r10240_ctx14336/summary.json`，即 14B SFT 合并模型，结果为答案正确 53.17%、严格 `real_pred` 22.00%。

`result.md` 中标为“本次”的两组结果位于 `outputs/test_result`，不在 `outputs/evaluation`：

- **本次 3B（8192 上限）**：`outputs/test_result/qwen_3b_8192`，对应 `outputs/sft/qwen3b` 模型目录；该目录的 README 明确说明它是基于 Qwen2.5-3B-Instruct 的 fine-tuned 模型，训练数据为 `graph_data`，因此是 SFT 模型。
- **本次 3B（16384 上下文）**：`outputs/test_result/qwen_3b`，同样对应 `outputs/sft/qwen3b`，只是推理 context 设置改为 16384；仍是同一个 SFT 模型，不是 RL 模型。
- **本次 14B**：`outputs/test_result/qwen_14b`，对应 `outputs/sft/qwen2.5_14b_lora_merged`；该模型 README 和 `supervised_finetuning` 配置明确表明它是 Qwen2.5-14B-Instruct 的 LoRA SFT 合并模型，不是 RL checkpoint。

## 5. 训练阶段的明确区分

| `result.md` 标签 | 实际训练阶段 | 是否经过 SFT | 是否经过 RL | 说明 |
|---|---|---:|---:|---|
| 3B（SFT，历史） | SFT | 是 | 否 | `outputs/sft/ds3b_lora_merged_20260822`；评测目录和报告均标为 LoRA SFT |
| 14B（SFT，历史） | SFT | 是 | 否 | `outputs/sft/qwen2.5_14b_lora_merged`；来源是 14B LoRA SFT |
| 3B（本次，8192 上限） | SFT | 是 | 否 | `outputs/sft/qwen3b`；“本次”只表示本次推理/汇总，不表示 RL |
| 3B（本次，16384 上下文） | SFT | 是 | 否 | 与上行相同的 SFT 模型，仅 context 配置不同 |
| 14B（本次） | SFT | 是 | 否 | `outputs/sft/qwen2.5_14b_lora_merged`；“本次”只表示本次推理/汇总 |

因此，`result.md` 中历史 3B、历史 14B、本次 3B、本次 14B **都经过了 SFT，且从现有 checkpoint 和日志证据看都没有经过 RL**。此前容易混淆的 1.5B 行才是 SFT 后继续进行 RL Stage 1 的模型；它来自 `outputs/rl/.../global_step_816`，不属于这里四个 SFT evaluation 目录。

## 6. 结论

`outputs/evaluation` 的四组结果在数据集、样本数和基本 SSR 审计目标上统一，但存在生成预算和模型 checkpoint 差异。3B 的主要瓶颈是超长、重复和截断；将上限从 8192 提高到 10240 只能小幅改善完整输出。14B SFT 在答案、结构完整性、EOS 和重复控制方面均明显更好。所有这些 evaluation 目录记录的都是 SFT 模型测试，不能把“历史/本次”理解成“历史 SFT/本次 RL”。
