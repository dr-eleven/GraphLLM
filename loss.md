• 结论先说：当前训练没有误把测试集当训练集，但也不能视为论文的等价复现。 数据文件大体属于正确任务，真正的问题是多项偏差叠加：

  1. 论文的 14B 全量训练被改成了 1.5B LoRA；
  2. RL 两阶段训练比例严重偏离论文；
  3. 奖励函数存在明确漏洞，会给空输出、循环和未闭合输出 0.05；
  4. 实际上下文上限只有 12,288，却配置了 16,384 response；
  5. RL prompt 被单独改写，与 SFT prompt 存在格式分布偏移；
  6. 发布仓库的 RL 数据域和难度划分本身就与论文不一致；
  7. Stage 1 曾有两个进程同时恢复同一 checkpoint，训练链路不够干净。

  这些因素共同解释了大量重复、response 截断以及正确率长期不升。

  ## 1. 数据集是否用对了

   数据                  审计结果                                                                                                                判断
  ━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   SFT 原始数据          8,265 条，数量与论文完全一致；训练域为 Instagram、WN18RR、Arxiv、Citeseer、Pubmed、Children、Computer、Photo、Sports    正确
  ────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────  ────────────────────────────────────────
   当前实际 SFT 数据     由于长度过滤，只保留 7,400 条，删除 865 条                                                                              不是错误数据，但不再等价于论文
  ────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────  ────────────────────────────────────────
   RL train              9,799 条，任务/标签/目标节点基本完整                                                                                    是发布仓库的训练集，但与论文描述不一致
  ────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────  ────────────────────────────────────────
   RL validation         FB15K237、Cora、Products、WikiCS 各 150 条，共 600 条                                                                   正确，无训练集泄漏
  ────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────  ────────────────────────────────────────
   strict SSR parquet    与原始 9,799 条的 dataset/task/label 完全相同，只修改了 prompt                                                          没换错数据，只改了 prompt

  SFT 原始文件 GraphSSR-master/outputs/synthetic_data/sft_data/graph_sft_data.json 的 8,265 个回答全部包含完整的 </think> 和 </Chosen_subgraph_reasoning>；用当前质量检查器重新验证，真实性、选择一致性和答案正确性也全部通过。同样的重复判据下，教师回答没有出现当前这种大面积行级循环。

  所以，重复并不是因为 SFT 标签本身包含大量重复文本。但教师轨迹非常长：原始平均约 6,171 completion tokens；过滤后仍约 5,749 tokens。这种长推理轨迹本来就是为论文的 14B 模型准备的。

  当前过滤配置见 GraphSSR-master/supervised_finetuning/lora_sft_1.5b.yaml:23 和过滤元数据 GraphSSR-master/outputs/synthetic_data/sft_data/graph_sft_data.filtered.lora_sft_1.5b.json.meta.json:1：

  - 8,265 → 7,400，删除 10.5%；
  - Link Prediction 删除 669/2,770，约 24.2%，明显高于其他任务；
  - 因此当前 SFT 不只是缩小规模，也改变了任务分布。

  ### RL 数据与论文的矛盾

  论文列出的训练域包括 Instagram 和 Citeseer，不包括 Reddit 和 History；但实际 RL 数据为：

  - WN18RR 2,000；
  - Reddit、Computer、Photo 各 1,000；
  - Arxiv 981、History 969、Sports 959、Pubmed 952、Children 938。

  这不是当前误操作导致的。原始发布仓库的 GraphSSR-master-ori/data_generation/pipelines/graph_prepare.py:13 和 GraphSSR-master-ori/README.md:93 就使用 Reddit、History，而论文附录 Table 9 使用 Instagram、Citeseer。也就是说：

  > 如果目标是复现“发布仓库”，现在的 RL train 路径基本正确；如果目标是严格复现论文，数据域不正确。

  另外，论文称 RL 数据是 10,000 条、Easy:Medium:Hard = 2:2:1，但发布数据只有 9,799 条。

  更重要的是，代码的难度划分也写错了。论文规定：

  - Easy：5 次中正确 4 或 5 次；
  - Medium：正确 2 或 3 次；
  - Hard：正确 0 或 1 次。

  原始代码 GraphSSR-master-ori/data_generation/pipelines/rl_data_generator.py:41 实际实现为：

  - Easy：正确率 ≥ 0.6，即 3/5、4/5、5/5；
  - Medium：1/5 或 2/5；
  - Hard：只有 0/5。

  因此一部分 3/5 样本被错误归为 Easy，1/5 被错误归为 Medium。这是发布代码与论文的实质性不一致，而且导出的 parquet 又丢弃了 difficulty 和 sample_index 字段，GraphSSR-master-ori/data_generation/pipelines/rl_data_generator.py:189 使现有 9,799 条数据无法完整追溯和重新核验难度比例。

  ## 2. Prompt 是否符合论文

  原始 prompt 的语义与论文 SSR 流程基本一致：

  - 采样 5 个候选子图；
  - 保留全部中心节点；
  - 不得虚构节点和边；
  - 至少一个中心节点-only子图；
  - 候选子图应多样；
  - 选择一个子图；
  - “repeat and verify”一次；
  - 只根据选中子图推理。

  对应原始模板在 GraphSSR-master-ori/data_generation/pipelines/prompt_templete.py:20，与论文及附录 (https://arxiv.org/abs/2603.02938)一致。

  但当前存在一个格式偏移：

  - SFT 的 8,265 条 prompt 全部使用抽象的 <Subgraph_i> 模板；
  - RL strict 数据使用显式的 <Subgraph_0> 至 <Subgraph_4> 五个完整模板块，并添加“必须恰好输出5块”的要求；
  - 这一修改来自 GraphSSR-master/scripts/upgrade_rl_prompts.py:25，每个 prompt 平均增加约 338 tokens。

  这个改动语义上是合理的，甚至有助于约束格式，但它不是论文/原始发布流程中的同格式训练：模型在 SFT 阶段从未见过这种显式五块 prompt，进入 RL 后输入格式突然变化。

  这不是重复的首要原因，但对 1.5B 模型很可能是中等强度的诱因。论文里的 “repeat and verify” 只要求复述选中子图一次，不能解释成千上万 token 的循环。

  ## 3. 最关键的代码问题：奖励函数错误奖励坏输出

  当前奖励函数与原始发布仓库相同，因此这是上游代码问题，不是最近修改引入的。

  在 GraphSSR-master/reinforcement_learning/verl/utils/reward_score/subgraph_size.py:115 中：

  status = True
  for i in range(len(sampled_subgraph_list)):
      ...
  return status

  如果 response 没有任何完整闭合的 <Subgraph_i>...</Subgraph_i>，解析结果是空列表，循环一次也不执行，status 仍然为 True。这就是典型的“空集合真值”漏洞。

  我直接用当前奖励代码验证：

  - 空字符串：score=0.05
  - 只有循环文本：score=0.05
  - 未闭合的 <Chosen_subgraph_reasoning>：score=0.05

  而论文的奖励定义明确规定：不存在真实有效子图时 Status_real=False，奖励应当是 0。

  其他漏洞还包括：

  - 不检查是否恰好有 5 个候选；
  - 不检查编号是否为 0–4 且唯一连续；
  - 不检查 EOS、XML 是否闭合；
  - 没有重复惩罚和截断惩罚；
  - 重复候选子图也可以通过；
  - 邻居是否真正连接、边端点是否列入节点集合没有完整验证；
  - Chosen_subgraph=-1 可绕过越界判断并使用 Python 负索引；
  - Stage 2 的 size reward 只统计中心节点数+邻居数，不考虑边或语义有效性。

  这与日志高度吻合：

  - Stage 1 抽查的 202 个未闭合输出中，199 个获得 0.05；
  - 114 个高重复输出全部获得 0.05。

  因此模型进入循环后没有得到明确负反馈。GRPO 中 n=3 又较小，当同一 prompt 的三个 rollout 都落在相近的 0.05 平台时，组内优势接近零，几乎学不到如何退出循环。

  ## 4. Response 为什么大量截断

  当前脚本同时配置了：

  - max_prompt_length=4096
  - max_response_length=16384
  - vLLM max_model_len=12288

  见 GraphSSR-master/reinforcement_learning/stage1_lora.sh:53 和 GraphSSR-master/reinforcement_learning/stage1_lora.sh:93。

  这三个值不可能同时成立。实际约束是：

  prompt tokens + response tokens ≤ 12288

  例如 prompt 为 2,500 tokens 时，真正可生成的 response 只有约 9,788 tokens，而不是 16,384。

  日志中未闭合回答的 prompt+response 总长度集中在约 12,280–12,287，直接证明它们撞上的是 vLLM 上下文硬上限。Stage 1 打印样本中未闭合率 67.3%，Stage 2 为约 70%。

  日志仍显示 response_length/clip_ratio=0.0，是因为指标代码拿 response 长度与 16,384 宽度的 response tensor 比较，而不是与 vLLM 实际可用长度比较，见 GraphSSR-master/reinforcement_learning/verl/trainer/ppo/metric_utils.py:110。因此监控没有报告真实截断。

  更准确的因果关系是：

  > 1.5B 模型先进入局部重复循环 → 奖励没有处罚 → 一直生成 → 最后撞到 12,288 上下文限制 → XML 和答案被截断。

  上下文上限是截断的直接原因，但不是模型开始重复的初始原因。

  ## 5. 训练参数和论文偏差很大

  论文使用 DeepSeek-R1-Distill-Qwen2.5-14B，全参数 SFT；SFT 3 epoch、LR 1e-5、总 batch 64；RL 总计 4 epoch、LR 1e-6、batch 64，且只有最后 2,048 个训练实例启用 Stage 2 去噪奖励。论文 (https://arxiv.org/pdf/2603.02938)

  当前配置为：

   项目                         论文            当前
  ━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━
   模型                          14B            1.5B
  ────────────────  ─────────────────  ──────────────
   微调                         Full    LoRA rank 32
  ────────────────  ─────────────────  ──────────────
   SFT 数据                    8,265           7,400
  ────────────────  ─────────────────  ──────────────
   SFT LR                       1e-5            1e-4
  ────────────────  ─────────────────  ──────────────
   SFT 有效 batch                 64            约 6
  ────────────────  ─────────────────  ──────────────
   RL LR                        1e-6            3e-6
  ────────────────  ─────────────────  ──────────────
   GRPO rollout n           原仓库 5               3
  ────────────────  ─────────────────  ──────────────
   PPO minibatch                  64               4
  ────────────────  ─────────────────  ──────────────
   KL coef              原仓库 0.001            0.02
  ────────────────  ─────────────────  ──────────────
   Stage 1            总训练的大部分         2 epoch
  ────────────────  ─────────────────  ──────────────
   Stage 2           最后 2,048 实例    完整 2 epoch

  其中最严重的是阶段划分。当前 Stage 2 约为：

  9778 × 2 ≈ 19556 个实例

  论文只在最后 2,048 个实例开启去噪奖励，当前约多了 9.5 倍。与此同时 Stage 1 真实性训练又明显不足。

  原仓库正是用 total_training_steps=2、每步 batch 1024 来实现最后 2,048 实例，见 GraphSSR-master-ori/reinforcement_learning/run_rl_stage2.sh:43。当前却设置 total_epochs=2，GraphSSR-master/reinforcement_learning/stage2_lora.sh:139。

  这非常能解释 Stage 2 没有改善反而波动下降：模型在尚未稳定学会真实、完整 SSR 输出时，就被长时间推动去选择小子图。

  ## 6. 模型规模是重复退化的主要基础因素

  论文明确选择 14B 是为了其更强的推理能力；论文还报告 GraphSSR 平均推理约 6,982 tokens。这是一个非常长的多阶段生成任务。

  当前却让 1.5B 模型完成：

  长图输入
  → 长 CoT
  → 构造 5 个不同子图
  → 记住每个子图
  → 选择一个
  → 精确复述
  → 分类并闭合 XML

  这远超一般 1.5B 模型稳定执行长程状态跟踪的能力。长文本生成中一旦进入高概率局部模式，temperature=0.5、top_p=0.9 和很弱的 repetition_penalty=1.03 很难跳出；验证阶段还是 greedy decoding，更容易稳定复现循环。

  SFT 的训练 loss 为 0.5297，但没有验证集、没有 SFT-only SSR 完整率评估，GraphSSR-master/outputs/sft/ds1.5b_lora/trainer_state.json:1。低训练 loss 只能说明拟合 token，不能证明它学会了长程 SSR 状态机。

  ## 7. 当前性能应如何判断

  Stage 1 最佳 checkpoint step 800：

  - answer accuracy：13.0%
  - 完整真实且正确的 real_pred：7.5%
  - reward：0.1197

  最终 step 814：

  - answer accuracy：10.33%
  - real_pred：5.17%

  Stage 2 前 300 步仍在约 11%–13% answer accuracy、6%–7% real_pred 区间，没有稳定改善。论文相应结果是约 68%–79% 区间，因此当前不是轻微复现偏差，而是训练目标尚未学成。

  而且当前“best checkpoint”按普通答案 pred 选择，不按完整 SSR 的 real_pred 选择：GraphSSR-master/reinforcement_learning/verl/trainer/ppo/ray_trainer.py:666。所以一个猜对答案、但子图结构损坏的 checkpoint 也可能被选为最佳模型。

  还有一项运行完整性风险：Stage 1 日志显示 PID 871836 和 PID 869681 几乎同时从同一个 global_step_350 恢复，GraphSSR-master/outputs/rl/1.5b_lora/verl_grpo_graph_rl_stage1_lora/rl1_lora.log:445766。两个进程共享输出/checkpoint 目录，存在 optimizer、latest/best 元数据或 checkpoint 相互覆盖的风
  险。不能证明最终 step 800 已损坏，但这条训练链不应作为严格复现实验依据。

  ## 根因优先级

  按我对重复与截断现象的贡献排序：

  1. 1.5B 模型承担为 14B 设计的超长 SSR 轨迹；
  2. 奖励漏洞把空、循环、未闭合输出奖励为 0.05；
  3. 12,288 实际上下文与 16,384 response 配置冲突；
  4. Stage 1 太短、Stage 2 约多训练 9.5 倍；
  5. GRPO n=3、minibatch 4、稀疏奖励导致学习信号弱且噪声大；
  6. SFT/strict-RL prompt 格式不一致；
  7. RL 数据域与难度划分不符合论文；
  8. 数据中的少量重复边、自环等序列化问题——有影响，但不是主要原因。

  ## 建议的修复顺序

  1. 先修奖励：空候选必须为 false；严格要求 5 个闭合且唯一的候选；禁止负索引；增加 EOS、截断、重复和结构连通性检查。
  2. 用独立新目录重新开始干净训练，禁止两个进程共享 checkpoint 目录。
  3. Stage 2 在 batch 24 下只运行约 ceil(2048/24)=86 步，而不是 814 步；其余预算留给 Stage 1。
  4. 对齐长度：在 12,288 上下文下使用 4096+8192，或者把 max_model_len 提高到至少 20,480；同时正确记录 vLLM finish reason。
  5. 恢复 LR≈1e-6、n=5，尽可能增大 PPO minibatch；KL 可从原始仓库的 0.001 附近重新试验。
  6. SFT 和 RL 使用完全相同的 prompt 格式。要么都使用原始抽象模板，要么重新生成显式五块格式的 SFT 数据。
  7. 若要求论文级复现，应使用 14B；若必须使用 1.5B，应缩短推理轨迹或只蒸馏最终结构化 SSR 输出，不能期待直接达到论文性能。
  8. 若要求数据严格一致，应重新生成包含 Instagram、Citeseer 的 RL 数据，并修正 4/5、2/3、0/1 的难度阈值，同时保留 difficulty/sample_index。
  9. RL 前先单独评测 merged SFT：完整闭合率、重复率、真实子图率和 answer accuracy。否则无法区分问题是在 SFT 已经形成，还是由 RL 放大。

## 8. 三类样本原文对照（2026-08-22）

下面三例不是挑选的极端个案，而是按各自组的中位数选取：SFT 样本取 `completion_tokens` 最接近中位数 5934 的记录；482 条截断结果取 `max_32gram_repeat` 最接近该组中位数 70 的记录；118 条正常停止结果取 `response_tokens` 最接近该组中位数 5310、同时结构完整的记录。为避免把数万 token（尤其是机械重复内容）直接灌入本文，只保留可复核的原文关键段；省略处均明确标出，完整原文可按文件路径和数组/评测索引读取。

### 8.1 SFT 数据集样本：中位长度的教师答案

- 原始文件：`outputs/synthetic_data/sft_data/graph_sft_data.filtered.lora_sft_3b.json`
- JSON 数组下标：`3957`
- 原始 `index/sample_index`：`5107/5107`
- 数据集/任务：`computer/lp`
- 标签：`No, they do not have co-purchased or co-viewed relationships`
- `completion_tokens`：`5934`（过滤后 SFT 数据的中位数）
- 字符长度：prompt `7441`，response `21646`

Prompt 原文节选：

```text
Task Description:
You are a graph analysis expert. Given a complete subgraph structure, which includes the description of the central node(s), its neighboring nodes, and their connection relationships, please follow the following steps to operate the analysis:

Complete subgraph:
- Central node ID(s): node0, node1
- Node texts (central node(s) and neighboring nodes): [
  node0:Highly positive review of HP TFT display enhancing Windows Vista Aero experience with sharp visuals and color accuracy.
  node1:Positive feedback on replacement strap's quick shipping and functionality.
  node2:Durable HP monitor praised for dual-screen setups and adjustable stand in office environments.
  node3:Brief positive note on satisfaction with product as expected.
  node4:Cost-effective HP monitors with reliability and crisp display for office multi-monitor setups.
  node5:High-quality monitor with good graphics and value, recommended purchase.
  node6:HP monitor praised for design and display but critic

[... 中间省略 5141 个字符；完整原文按上方路径和索引读取 ...]

e separate them with commas</Central_node_ID>
  <Neighboring_node_ID>the neighboring node ids, separate them with commas, if it does not exist, leave it blank</Neighboring_node_ID>
  <Connection_relationship>the connection relationships, each relation is presented in the form of <node_ID, node_ID>, separated by commas, if it does not exist, leave it blank</Connection_relationship>
  <Sampling_reason>the reason for sampling this subgraph</Sampling_reason>
</Subgraph_i>
<Chosen_subgraph_reasoning>
  <Chosen_subgraph>directly give the index number of the chosen subgraph</Chosen_subgraph>
  <Chosen_subgraph_reason>the reason for choosing this subgraph</Chosen_subgraph_reason>
  <Central_node_ID>the central node id(s), if there are multiple ones, please separate them with commas</Central_node_ID> 
  <Neighboring_node_ID>the neighboring node ids, separate them with commas, if it does not exist, leave it blank</Neighboring_node_ID>
  <Connection_relationship>the connection relationships, each relation is presented in the form of <node_ID, node_ID>, separated by commas, if it does not exist, leave it blank</Connection_relationship>
  <Bond_value>the bond value</Bond_value>
  <Answer>the answer</Answer>
  <Brief_reasoning>the brief reasoning</Brief_reasoning>
</Chosen_subgraph_reasoning>

```

Response 原文节选（保留开头及完整最终 XML）：

```text
<think>
We are given a complete subgraph with central nodes node0 and node1, and a set of node texts and connection relationships.

Step 1: Subgraph Sampling
We need to sample 5 subgraphs that:
- Always include the central nodes (node0 and node1).
- Can include none, some, or all of the one-hop or two-hop neighbors.
- One subgraph must be only the central nodes (node0 and node1).
- One subgraph should contain common neighboring nodes that connect the two central nodes (if such nodes exist). If not, we don't fabricate.
- The subgraphs should be distinct and not similar, with significant information differences and varying sizes.

Let's note the neighbors of the central nodes from the connection relationships:

Connections involving node0 (from the list):
- <node0, node6>
- <node0, node5>
- <node0, node4>
- <node0, node2>
- <node0, node3>

Connections involving node1:
- <node7, node1>  (wh

[... 中间省略 16988 个字符 ...]

e, they are not co-purchased or co-viewed.

But note: the bond value is a number. We set it to 0.0 because we see no positive signal.

Alternatively, we could set it to a low value. But the threshold is 0.5, so any value below 0.5 would lead to "No". We can set it to 0.0.

Let me double-check: in the chosen subgraph, is there any edge that might connect the two central nodes? No. And the products are different.

Therefore, we output accordingly.
</think>
<Subgraph_0>
  <Central_node_ID>node0,node1</Central_node_ID>
  <Neighboring_node_ID></Neighboring_node_ID>
  <Connection_relationship></Connection_relationship>
  <Sampling_reason>This minimal subgraph contains only the central nodes, providing a baseline for isolated node analysis without neighbor influence.</Sampling_reason>
</Subgraph_0>
<Subgraph_1>
  <Central_node_ID>node0,node1</Central_node_ID>
  <Neighboring_node_ID>node2,node3,node4,node5,node6,node7,node8</Neighboring_node_ID>
  <Connection_relationship><node0,node2>,<node0,node3>,<node0,node4>,<node0,node5>,<node0,node6>,<node1,node7>,<node1,node8></Connection_relationship>
  <Sampling_reason>Includes all direct neighbors of both central nodes, capturing comprehensive first-order connections and semantic diversity (monitors vs. straps).</Sampling_reason>
</Subgraph_1>
<Subgraph_2>
  <Central_node_ID>node0,node1</Central_node_ID>
  <Neighboring_node_ID>node6,node8,node24</Neighboring_node_ID>
  <Connection_relationship><node0,node6>,<node1,node8>,<node24,node8></Connection_relationship>
  <Sampling_reason>Focuses on neighbors with negative feedback (node6, node24) and their connections, contrasting with positive nodes in other subgraphs.</Sampling_reason>
</Subgraph_2>
<Subgraph_3>
  <Central_node_ID>node0,node1</Central_node_ID>
  <Neighboring_node_ID>node5,node17,node18</Neighboring_node_ID>
  <Connection_relationship><node0,node5>,<node17,node5>,<node5,node18></Connection_relationship>
  <Sampling_reason>Extends to two-hop neighbors (node17, node18) from node0's positive connections, emphasizing monitor-centric satisfaction.</Sampling_reason>
</Subgraph_3>
<Subgraph_4>
  <Central_node_ID>node0,node1</Central_node_ID>
  <Neighboring_node_ID>node7,node21,node22,node23,node24</Neighboring_node_ID>
  <Connection_relationship><node1,node7>,<node7,node21>,<node22,node7>,<node23,node8>,<node24,node8></Connection_relationship>
  <Sampling_reason>Centers on node1's strap-related ecosystem with deeper neighbor chains, highlighting durability complaints and bag accessories.</Sampling_reason>
</Subgraph_4>
<Chosen_subgraph_reasoning>
  <Chosen_subgraph>1</Chosen_subgraph>
  <Chosen_subgraph_reason>This subgraph includes all direct neighbors of both central nodes, maximizing contextual information about their respective product domains (monitors vs. straps) without relying on unobserved connections. It provides the broadest semantic coverage for link prediction.</Chosen_subgraph_reason>
  <Central_node_ID>node0,node1</Central_node_ID>
  <Neighboring_node_ID>node2,node3,node4,node5,node6,node7,node8</Neighboring_node_ID>
  <Connection_relationship><node0,node2>,<node0,node3>,<node0,node4>,<node0,node5>,<node0,node6>,<node1,node7>,<node1,node8></Connection_relationship>
  <Bond_value>0.0</Bond_value>
  <Answer>No, they do not have co-purchased or co-viewed relationships</Answer>
  <Brief_reasoning>Node0 (monitor) and node1 (strap) have entirely disjoint neighbor sets: node0's neighbors are all monitor-related (node2, node3, node4, node5, node6), while node1's neighbors are strap-related (node7, node8). No shared neighbors or semantic overlap exists in this subgraph to suggest co-purchasing/viewing relationships between such dissimilar products.</Brief_reasoning>
</Chosen_subgraph_reasoning>
```

这条教师数据能正常完成 `</think>`、5 个候选子图及 `</Chosen_subgraph_reasoning>`，而且答案与标签一致。它也直观显示了训练目标为何很重：总计 5934 个 completion token，其中绝大部分用于自然语言规划、枚举、复核，最终结构化 XML 只占末段。

### 8.2 482 条截断结果中的代表性重复样本

- 推理输入：`outputs/synthetic_data/rl_data/test_600_graphr2_subgraph_strict_ssr.parquet`，行号/评测索引 `181`
- 推理输出：`outputs/evaluation/ds3b_lora_sft_20260822_ssr_600_r10240_ctx14336/shard-01-of-03.jsonl`，`index=181`
- 审计记录：同目录 `audit_rows.jsonl`，`index=181`
- 数据集/任务：`wikics/nc`
- 正确标签：`distributed computing architecture`
- token：prompt `3068`，response `10240`
- 停止原因：`finish_reason=length`
- 重复指标：`max_line_repeat=71`，`duplicate_line_fraction=0.4259818731`，`max_32gram_repeat=70`（截断组中位水平）
- 结构结果：没有 `</think>`，没有进入最终 XML，答案为空，`xml_complete=false`

Prompt 原文节选：

```text
Task Description:
You are a graph analysis expert. Given a complete subgraph structure, which includes the description of the central node(s), its neighboring nodes, and their connection relationships, please follow the following steps to operate the analysis:

Complete subgraph:
- Central node ID(s): node26
- Node texts (central node(s) and neighboring nodes): [
  node0: Amazon Elastic Block Store (EBS) offers EC2-linked block storage with SSD/disk options, snapshotting, and scalability, launched in 2008 with a free tier.
  node1: Amazon Mechanical Turk is a crowdsourcing platform for human intelligence tasks, used for data labeling and research, criticized for low worker pay and lack of labor protections.
  node2: ShareFile provides secure file-sharing/collaboration tools, offering cloud/on-prem storage, client portals, and compliance features post-Citrix acquisition.
  node3: Cloud co

[... 中间省略 9914 个字符；完整原文按上方路径和索引读取 ...]

son>[non-empty sampling reason]</Sampling_reason>
</Subgraph_3>
<Subgraph_4>
  <Central_node_ID>[central node ID(s)]</Central_node_ID>
  <Neighboring_node_ID>[neighboring node ID(s), or blank]</Neighboring_node_ID>
  <Connection_relationship>[edges, or blank]</Connection_relationship>
  <Sampling_reason>[non-empty sampling reason]</Sampling_reason>
</Subgraph_4>
<Chosen_subgraph_reasoning>
  <Chosen_subgraph>directly give the index number of the chosen subgraph</Chosen_subgraph>
  <Chosen_subgraph_reason>the reason for choosing this subgraph</Chosen_subgraph_reason>
  <Central_node_ID>the central node id(s), if there are multiple ones, please separate them with commas</Central_node_ID> 
  <Neighboring_node_ID>the neighboring node ids, separate them with commas, if it does not exist, leave it blank</Neighboring_node_ID>
  <Connection_relationship>the connection relationships, each relation is presented in the form of <node_ID, node_ID>, separated by commas, if it does not exist, leave it blank</Connection_relationship>
  <Answer>the answer</Answer>
  <Brief_reasoning>the brief reasoning</Brief_reasoning>
</Chosen_subgraph_reasoning>

```

Response 原文节选（开头、实际循环单元和被硬截断的结尾）：

```text
<think>
We are given a complete subgraph with central node ID: node26.
 The task is to sample 5 subgraphs for node classification of the central node (node26). The subgraphs must include the central node(s) (node26) and can include no, partial, or all one-hop or two-hop neighbors.

 Steps:
 1. Sample 5 subgraphs that are distinct and have significant information differences. One subgraph must be only the central node. Another should include as many neighbors as possible that semantically support node26.
 2. The subgraphs should be as different in structure and size as possible.

 Let's note the central node: node26. Its text: "Amazon Drive is a cloud storage service offering free photo storage for Prime members, with plans, device compatibility, and features like file sharing and photo printing."

 We are to classify node26. The options are given.

 First, we note the complete graph has 31 nodes (node0 to node30) and the connections provided.

 We must sample 5 subgraphs:

 Subgraph 0: Only the central node (node26). This is required.

 Subgraph 1: We want one that includes as many n

[... 首次进入循环前的推理省略 ...]

 But we can also have:

 Subgraph 2: [node26, node2, node4, node12, node13, node14, node16, node19, node24, node26, node27, node29] (same as above) -> but we want distinct. So we can have a different one.

 But we can also have:

 Subgraph 2: [node26, node2, node4, node12, node13, node14, node16, node19, node24, node26, node27, node29] (same as above) -> but we want distinct. So we can have a different one.


[... 上述两行模式持续重复，直至达到生成上限 ...]

4, node12, node13, node14, node16, node19, node24, node26, node27, node29] (same as above) -> but we want distinct. So we can have a different one.

 But we can also have:

 Subgraph 2: [node26, node2, node4, node12, node13, node14, node16, node19, node24, node26, node27, node29] (same as above) -> but we want distinct. So we can have a different one.

 But we can also have:

 Subgraph 2: [node26, node2, node4, node12, node13, node14, node16, node19, node24, node26, node27, node29] (same as above) -> but we want distinct. So we can have a different one.

 But we can also have:

 Subgraph 2: [node26, node2, node4, node12, node13, node14, node16, node19, node24, node26, node27, node29] (same as above) -> but we want distinct. So we can have a different one.

 But we can also have:

 Subgraph 2: [node26, node2, node4, node12, node13, node14, node16, node19, node24, node26, node27, node29] (
```

这条样本先进行了看似正常的子图规划，随后在“需要不同的 Subgraph 2”这一局部状态中失去进展。同一个候选节点列表行出现 71 次，相邻的 `But we can also have:` 出现 70 次；模型没有生成 `</think>` 或任何最终 XML，最终由 10240 的生成上限强制截断。它代表的不是“答案太长所以刚好少了结尾”，而是长思考阶段进入循环后耗尽全部 token。

### 8.3 118 条正常停止结果中的代表样本

- 推理输入：`outputs/synthetic_data/rl_data/test_600_graphr2_subgraph_strict_ssr.parquet`，行号/评测索引 `507`
- 推理输出：`outputs/evaluation/ds3b_lora_sft_20260822_ssr_600_r10240_ctx14336/shard-00-of-03.jsonl`，`index=507`
- 审计记录：同目录 `audit_rows.jsonl`，`index=507`
- 数据集/任务：`products/nc`
- 正确标签：`Tools & Home Improvement`
- token：prompt `2176`，response `5310`（正常停止组中位数）
- 停止原因：`finish_reason=stop`
- 重复指标：`max_32gram_repeat=1`，`high_repetition=false`
- 结构与答案：`xml_complete=true`、恰好 5 个候选子图、存在中心节点-only子图、选择一致、答案正确
- 严格图真实性：`strict_real_subgraphs=false`，因此 `real_pred=false`；这说明“正常停止且答案正确”仍不等同于所有输出边都通过严格真实性审计

Prompt 原文节选：

```text
Task Description:
You are a graph analysis expert. Given a complete subgraph structure, which includes the description of the central node(s), its neighboring nodes, and their connection relationships, please follow the following steps to operate the analysis:

Complete subgraph:
- Central node ID(s): node11
- Node texts (central node(s) and neighboring nodes): [
  node0: Bosch 13-piece carrying case with insert set for Click & Go System tools.
  node1: Bosch 18-piece titanium twist drill bit set in plastic case.
  node2: Watts 10-pack lead-free brass PEX elbows for potable systems; installs with Cinch tools or crimp rings; NSF/ASTM certified.
  node3: Bosch 13-piece spade bit set (Daredevil DSB5013P).
  node4: Drill Doctor 750X sharpens 3/32-3/4" bits with diamond wheel; includes case, chucks, and instructional DVD.
  node5: Bosch 7-amp 1/2-inch VSR hammer drill with 360-degree handle,

[... 中间省略 6634 个字符；完整原文按上方路径和索引读取 ...]

son>[non-empty sampling reason]</Sampling_reason>
</Subgraph_3>
<Subgraph_4>
  <Central_node_ID>[central node ID(s)]</Central_node_ID>
  <Neighboring_node_ID>[neighboring node ID(s), or blank]</Neighboring_node_ID>
  <Connection_relationship>[edges, or blank]</Connection_relationship>
  <Sampling_reason>[non-empty sampling reason]</Sampling_reason>
</Subgraph_4>
<Chosen_subgraph_reasoning>
  <Chosen_subgraph>directly give the index number of the chosen subgraph</Chosen_subgraph>
  <Chosen_subgraph_reason>the reason for choosing this subgraph</Chosen_subgraph_reason>
  <Central_node_ID>the central node id(s), if there are multiple ones, please separate them with commas</Central_node_ID> 
  <Neighboring_node_ID>the neighboring node ids, separate them with commas, if it does not exist, leave it blank</Neighboring_node_ID>
  <Connection_relationship>the connection relationships, each relation is presented in the form of <node_ID, node_ID>, separated by commas, if it does not exist, leave it blank</Connection_relationship>
  <Answer>the answer</Answer>
  <Brief_reasoning>the brief reasoning</Brief_reasoning>
</Chosen_subgraph_reasoning>

```

Response 原文节选（保留开头、`</think>` 前结论及完整最终 XML）：

```text
<think>
We are given a complete subgraph with central node: node11.
 The task is to sample 5 subgraphs that must include the central node(s) (which is node11). The subgraphs can include no, partial, or all one-hop or two-hop neighbors.

 Steps for sampling:
 1. There must be a subgraph that consists only of the central node (node11).
 2. There should be a subgraph that includes as many neighboring nodes as possible that semantically support the central node. We must check if such a subgraph exists (i.e., if there are neighbors that are related to the central node's category).
 3. The subgraphs should be distinct and have different structural and semantic information, with varying sizes.

 Let's note the central node: node11 is "Hitachi 6-amp 3/8-in drill with keyless chuck, variable speed, ergonomic grip; 108.1 in-lb torque, 3.1 lbs."

 The complete graph has the following nodes and edge

[... 中间省略 14114 个字符 ...]

ch is a different tool.

 The central node (node11) is a drill (a tool). The neighbors are all tools (drill bits, sharpener, hammer drill, impact driver, screwdriver). Therefore, the central node is a tool.

 Looking at the options, the category that fits is "Tools & Home Improvement".

 So the answer is: Tools & Home Improvement

 Brief reasoning: The central node (node11) is a drill, and all the neighboring nodes in the chosen subgraph are tools (drill bits, sharpener, hammer drill, impact driver, screwdriver). Therefore, the central node belongs to the category "Tools & Home Improvement".

 Now, we write the answer in the required format.
</think>
<Subgraph_0>
  <Central_node_ID>node11</Central_node_ID>
  <Neighboring_node_ID></Neighboring_node_ID>
  <Connection_relationship></Connection_relationship>
  <Sampling_reason>Sampled to analyze the central node in isolation, providing a baseline for classification without neighbor influence.</Sampling_reason>
</Subgraph_0>
<Subgraph_1>
  <Central_node_ID>node11</Central_node_ID>
  <Neighboring_node_ID>node1</Neighboring_node_ID>
  <Connection_relationship><node1, node11></Connection_relationship>
  <Sampling_reason>Includes one direct neighbor (node1) that shares a similar tool category (drill bit set) to test minimal contextual support.</Sampling_reason>
</Subgraph_1>
<Subgraph_2>
  <Central_node_ID>node11</Central_node_ID>
  <Neighboring_node_ID>node1, node3</Neighboring_node_ID>
  <Connection_relationship><node1, node11>, <node1, node3></Connection_relationship>
  <Sampling_reason>Combines two drill-related neighbors (node1 and node3) to explore small-group context while maintaining structural simplicity.</Sampling_reason>
</Subgraph_2>
<Subgraph_3>
  <Central_node_ID>node11</Central_node_ID>
  <Neighboring_node_ID>node1, node10</Neighboring_node_ID>
  <Connection_relationship><node1, node11>, <node1, node10></Connection_relationship>
  <Sampling_reason>Uses a different pair of neighbors (node10) to contrast with Subgraph_2, emphasizing drill-bit specificity.</Sampling_reason>
</Subgraph_3>
<Subgraph_4>
  <Central_node_ID>node11</Central_node_ID>
  <Neighboring_node_ID>node1, node3, node4, node5, node6, node8, node10, node12</Neighboring_node_ID>
  <Connection_relationship><node1, node11>, <node1, node3>, <node1, node4>, <node1, node5>, <node1, node6>, <node1, node8>, <node1, node10>, <node1, node12></Connection_relationship>
  <Sampling_reason>Maximizes semantically supportive neighbors (all drill-related nodes) to capture comprehensive tool context for robust classification.</Sampling_reason>
</Subgraph_4>
<Chosen_subgraph_reasoning>
  <Chosen_subgraph>4</Chosen_subgraph>
  <Chosen_subgraph_reason>This subgraph includes all drill-related neighbors (node1, node3, node4, node5, node6, node8, node10, node12) that directly support the central node's tool classification, providing the richest contextual information.</Chosen_subgraph_reason>
  <Central_node_ID>node11</Central_node_ID>
  <Neighboring_node_ID>node1, node3, node4, node5, node6, node8, node10, node12</Neighboring_node_ID>
  <Connection_relationship><node1, node11>, <node1, node3>, <node1, node4>, <node1, node5>, <node1, node6>, <node1, node8>, <node1, node10>, <node1, node12></Connection_relationship>
  <Answer>Tools & Home Improvement</Answer>
  <Brief_reasoning>The central node (node11) is a drill. All neighboring nodes in this subgraph are tools (drill bits, sharpener, hammer drill, impact driver, screwdriver) that fall under the "Tools & Home Improvement" category. No other options fit the context.</Brief_reasoning>
</Chosen_subgraph_reasoning>
```

与截断例相比，这条记录能从采样规划推进到类别判断，主动闭合 `</think>`，然后一次性输出完整结构并由 EOS 正常停止。两者 prompt/response 规模并没有数量级差异，关键区别是正常样本没有陷入局部重复；这进一步支持“482 条截断的主因是思考阶段循环退化，而不是 10240 本身普遍容纳不下正常答案”。



• 结论：14B 的 600 条推理已经全部完成，但原验证流程没有完成最后的审计落盘。当前没有验证进程在运行，结果文件包含唯一索引 0–599；我用相同审计脚本重新计算成功，因此模型效果已经可以判断。

  ### 14B vs 3B

   指标                    3B             14B                    变化
  ━━━━━━━━━━━━━━━━━━  ━━━━━━━━  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━
   答案正确率          13.50%          53.17%               +39.67 pp
  ──────────────────  ────────  ──────────────  ──────────────────────
   完整 XML            19.83%          73.50%               +53.67 pp
  ──────────────────  ────────  ──────────────  ──────────────────────
   五个子图全部真实     5.00%          30.67%               +25.67 pp
  ──────────────────  ────────  ──────────────  ──────────────────────
   严格 real_pred       3.33%          22.00%    +18.67 pp，约 6.6 倍
  ──────────────────  ────────  ──────────────  ──────────────────────
   长度截断            80.33%          26.33%               -54.00 pp
  ──────────────────  ────────  ──────────────  ──────────────────────
   高重复              79.67%          26.33%               -53.34 pp
  ──────────────────  ────────  ──────────────  ──────────────────────
   平均输出长度         9,323    5,130 tokens                 -44.98%

  同一批样本的成对比较也很明确：

  - 14B 新增答对 261 条，退步 23 条，二者都答对 58 条。
  - 14B 新增严格成功 123 条，退步 11 条，二者都成功 9 条。
  - 因此提升不是由少量样本或统计波动造成的。

  ### 14B 当前效果分析

  14B 已经基本解决 3B 最严重的“无限重复直到截断”问题：

  - 442/600 正常停止，其中 441 条输出完整 XML。
  - 158 条被截断，并且这 158 条全部属于高重复输出。
  - 在正常停止的 442 条里，319 条答案正确，条件正确率为 72.17%。

  现在主要瓶颈从“无法结束”转成了“子图真实性/图结构约束”：

  - 441 条完整 XML 中，有 257 条未通过严格真实子图检查。
  - 只有 184 条五个子图全部真实；其中 132 条分类正确、52 条分类错误。
  - 即便只看正常结束结果，严格成功率也只有 29.86%，所以 14B 虽显著优于 3B，但绝对效果仍不够稳。

  分任务表现：

   数据集      3B→14B 答案正确率    3B→14B 严格成功率
  ━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━
   FB15K237       4.67% → 48.67%        1.33% → 6.00%
  ──────────  ───────────────────  ───────────────────
   Cora          11.33% → 52.00%       2.67% → 29.33%
  ──────────  ───────────────────  ───────────────────
   Products      22.00% → 54.00%       6.00% → 28.67%
  ──────────  ───────────────────  ───────────────────
   WikiCS        16.00% → 58.00%       3.33% → 24.00%

  FB15K237 仍明显最弱：虽然分类答案提升很大，但严格子图成功只有 6%，说明关系方向、边真实性或子图一致性仍有明显问题。

  ### 是否可以进入 RL

  暂时不建议直接使用当前 legacy reward 开始 RL：

  - legacy real_pred 为 24.83%，严格口径是 22.00%，仍有高估。
  - 158 条截断/重复输出仍会获得 0.05 奖励，存在强化坏行为的风险。
  - 更适合先修正 reward：截断或高重复直接记 0，并加强真实边、节点范围、五个子图差异性检查。

  另外，这不是纯粹的“参数规模”对照：14B 使用 8,192 cutoff、5,254 条训练样本和 2e-5 学习率；3B 使用 10,240 cutoff、7,387 条样本和 5e-5。所以应表述为“当前 14B 训练方案相对当前 3B 方案显著提升”，不能完全归因于模型规模。
