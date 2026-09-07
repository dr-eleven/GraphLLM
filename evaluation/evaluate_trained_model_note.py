"""
GraphSSR 模型评测入口模块 —— 在 GOFA 基准上评估训练好的图推理模型。

================================================================================
模块概览
================================================================================

本文件是 GraphSSR 项目训练完成后的最终评测入口。它不参与模型参数更新，
而是将训练好的模型作为"黑盒"进行零样本图推理能力测试。

================================================================================
核心功能
================================================================================

1. 数据加载与 prompt 重建
   - 从 GOFA 测试 CSV 中读取原始数据（prompt、ground_truth、数据集名、任务类型）
   - 用正则表达式从原始 prompt 中提取结构化信息（目标节点、邻居节点描述、边关系、候选标签）
   - 调用 construct_prompt_chosen_reasoning 将提取的信息重建为 GraphSSR 的 SSR 标准 prompt 格式

2. 模型推理
   - 将重建后的 prompt 发送到已部署的 vLLM 推理服务器（HTTP API）
   - 通过 utils/utils.py 中的 reason() 函数实现并发批量推理
   - 支持多次重试：如果格式合规率太低（错误样本超过 1/1000），自动重试最多 8 次

3. 答案提取与评估
   - 从模型响应中按 SSR 格式提取 <Answer> 标签内的答案
   - 将答案分类为四种类型：正确回答、错误回答、未知回答、格式错误
   - 按数据集和任务类型分别计算准确率和格式合规率
   - 计算每个标签类别级别的细粒度准确率

================================================================================
在训练流程中的角色
================================================================================

位置：训练后评估阶段（Post-training Evaluation）
前置：必须先用 scripts/deploy_api.sh 将训练好的模型部署为 vLLM HTTP 服务
输入：GOFA 测试 CSV + 部署好的 vLLM 推理端点
输出：按 (数据集, 任务类型) 组织的推理结果和评估指标 JSON 文件

================================================================================
命令行参数
================================================================================

-d, --data_path    : GOFA 测试 CSV 文件路径
-s, --save_dir     : 结果保存目录
-e, --seed         : 随机种子（默认 42）
-t, --temperature  : 推理温度参数（默认 0.5）
-c, --choose_server: 评测服务器列表（对应 URL_LIST_EVAL 的 key）
-g, --group_size   : 子图采样分组大小（默认 5）
"""

import re
import os
import sys
import json
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

# 将项目根目录加入 Python 搜索路径，以便导入项目内的工具模块
root_dir = ".."
sys.path.append(os.path.abspath(root_dir))

# =============================================================================
# 链接预测关系类型映射表
# =============================================================================
# 将数据集名称映射为该数据集中"边"所代表的自然语言关系描述。
# 在构造链接预测（LP）任务的 prompt 时，需要告诉模型"这两个节点之间是否存在某种关系"，
# 而这里的映射表提供了每种数据集对应的关系类型术语。
#
# 例如：
#   - arxiv / pubmed / cora：学术引用网络，边表示 citation（引用）关系
#   - children / computer / photo 等：Amazon 商品共现网络，边表示 co-purchased or co-viewed
#   - instagram：社交网络，边表示 following（关注）关系
#   - wikics：维基百科计算机科学文章网络，边表示 joint edited（共同编辑）关系
# =============================================================================
link_prediction_type_map = {
    # ---- 训练阶段使用的数据集 ----
    "arxiv": "citation",
    "pubmed": "citation",
    "children": "co-purchased or co-viewed",
    "history": "co-purchased or co-viewed",
    "computer": "co-purchased or co-viewed",
    "photo": "co-purchased or co-viewed",
    "sports": "co-purchased or co-viewed",
    "instagram": "following",
    "reddit": "replying",

    # ---- 评测阶段使用的数据集 ----
    "cora": "citation",
    "products": "co-purchased or co-viewed",
    "wikics": "joint edited",
    "cora_link": "citation",
}

# =============================================================================
# 评测推理服务端点配置
# =============================================================================
# 这是一个字典，key 是服务器的逻辑名称（如 "eval_server_0"），value 是连接参数。
# 实际使用时必须将以下占位符替换为真实值：
#   - YOUR_NODE  → 部署了 vLLM 服务的 GPU 节点 IP 或主机名
#   - PORT       → vLLM 服务的监听端口（与 deploy_api.sh 中的 PORT 一致）
#
# 各字段说明：
#   - name       : 服务器逻辑名称（用于日志和调试）
#   - model_name : 调用 API 时传入的 model 参数值（必须与 vLLM 的 --served-model-name 一致）
#   - pool_size  : 该服务器的并发请求池大小（控制同时发送的请求数量）
#   - url        : API 端点地址（OpenAI 兼容的 /v1/chat/completions 路径）
#   - status     : 服务器状态（"active" 表示可用）
# =============================================================================
URL_LIST_EVAL = {
    "eval_server_0": {
        "name": "eval_server_0",
        "model_name": "eval",
        "pool_size": 200,
        "url": "http://YOUR_NODE:PORT/v1/chat/completions",
        "status": "active"
    },
    # 可按需添加更多评估服务器端点（如需负载均衡或多模型对比评测）
}


# =============================================================================
# get_gofa_prompt() — GOFA 测试数据加载与 SSR prompt 重建
# =============================================================================
# 参数：
#   data_path : GOFA 测试集 CSV 文件路径
#   group_size: 子图采样时的分组大小（传递给 construct_prompt_chosen_reasoning）
#
# 返回值：
#   一个字典，key 为 "{dataset}+{task_type}" 格式的任务标识符，
#   value 为包含以下字段的字典：
#     - "messages"    : [[{"role": "user", "content": prompt}], ...] 格式的消息列表
#     - "truth_labels": 对应的标准答案列表
#     - "answer_set"  : 该任务下所有标准答案的去重集合
#
# 处理流程：
#   1. 用 pandas 加载 CSV 文件
#   2. 遍历每一行，用正则提取 prompt 中的结构化图信息
#   3. 跳过非 nc/lc 的任务（当前评测只覆盖节点分类和链接分类）
#   4. 调用 construct_prompt_chosen_reasoning 重建为标准 SSR prompt
#   5. 按 (数据集, 任务类型) 组织输出
# =============================================================================
def get_gofa_prompt(data_path, group_size):
    # 导入 SSR prompt 构造模板函数
    from data_generation.pipelines.prompt_templete import construct_prompt_chosen_reasoning

    # 加载 GOFA 测试 CSV 文件
    df = pd.read_csv(data_path)
    gofa_test_data = {}

    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"load test data"):
        # ---- 读取每行的基本字段 ----
        prompt = str(row['prompt'])              # 原始 prompt 文本
        dataset = str(row['dataset'])            # 数据集名称（如 cora_node, wikics）
        task_type = str(row['task'])             # 任务类型（nc/lp/lc/gc）
        truth_label = str(row['ground_truth'])   # 标准答案（ground truth）

        # ---- 从原始 prompt 中用正则提取目标节点 ----
        # 目标节点出现在 prompt 第一行，格式如 "node0 and node1"
        # 正则 r'node\d+' 匹配所有以 "node" 开头后跟数字的字符串
        target_node_pattern = r'node\d+'
        target_node_matches = re.findall(target_node_pattern, prompt.split("\n")[0], re.DOTALL)
        # 将节点列表用逗号+空格连接，如 "node0, node1"
        target_nodes = ', '.join([match.strip() for match in target_node_matches])

        # ---- 从原始 prompt 中用正则提取节点文本描述 ----
        # 每个节点一行，格式为 "nodeX: <文本描述>"
        # 正则 r'node\d+:.*?\n' 匹配从 nodeX: 开始到行尾的内容（非贪婪匹配）
        node_pattern = r'node\d+:.*?\n'
        node_matches = re.findall(node_pattern, prompt, re.DOTALL)
        # 每行前加两个空格缩进，用换行符连接
        node_information = '\n'.join(["  " + match.strip() for match in node_matches])

        # ---- 从原始 prompt 中用正则提取边关系 ----
        # 边的格式为 "<nodeX, nodeY>"，表示节点 X 和 Y 之间存在边
        edge_pattern = r'<node\d+, node\d+>'
        edge_matches = re.findall(edge_pattern, prompt, re.DOTALL)
        # 所有边用逗号+空格连接
        edge_information = ', '.join([match.strip() for match in edge_matches])

        # ---- 根据任务类型提取候选标签列表 ----
        if task_type == 'nc':
            # 节点分类：标签以 "Select strictly from: label1', 'label2' ..." 格式给出
            label_pattern = r'Select strictly from:(.*?)\.'
            matches = re.findall(label_pattern, prompt, re.DOTALL)
            # 根据是主测试集还是补充测试集，使用不同的分隔符
            if "gofa_test_data_53114.csv" in data_path:
                # 主测试集：标签用 "', '" 分隔
                label_list = [label.strip().strip("'") for label in matches[0].split("', '")]
            elif "gofa_supply_data_35603.csv" in data_path:
                # 补充测试集：标签用 ";" 分隔
                label_list = [label.strip() for label in matches[0].split(";")]
            else:
                raise ValueError("Unknown data_path for nc task")
            # 修复 cora_node 数据集中 "Theory" 单词的拼写错误
            if dataset == "cora_node" or dataset == "cora_node_2ways":
                label_list = [label.replace("Theroy", "Theory") for label in label_list]

        elif task_type == 'lc':
            # 链接分类：标签以 "Choose from the following: label1; label2; ..." 格式给出
            label_pattern = r'Choose from the following:(.*?)\.'
            matches = re.findall(label_pattern, prompt, re.DOTALL)
            label_list = [label.strip().strip("'") for label in matches[0].split(';')]

        elif task_type == 'lp':
            # 链接预测：只有两个选项（"有关系" 或 "无关系"）
            label_pattern = r'select strictly from: (.*?)\.Respond only with the'
            matches = re.findall(label_pattern, prompt, re.DOTALL)
            label_list = [label.strip().strip("'") for label in matches[0].split("' or '")]

        else:
            # 其他任务类型（如 gc）暂时不处理
            label_list = []

        # 将标签列表格式化为 "<label1>, <label2>, ..." 的形式
        label_list = "<" + ">, <".join(label_list) + ">"

        # ---- 当前只评测 nc（节点分类）和 lc（链接分类）任务，跳过其他 ----
        if row['task'] not in ["nc", "lc"]:
            continue

        # ---- 完整性检查：确保 prompt 中包含所有必要组件 ----
        if len(target_nodes) == 0:
            raise ValueError("target_nodes is missing in the prompt.")
        if len(node_matches) == 0:
            raise ValueError("node_matches is missing in the prompt.")
        elif len(node_matches) >= 2:
            # 如果有 2 个及以上节点但没有边信息，说明数据不完整
            if len(edge_matches) == 0:
                raise ValueError("edge_matches is missing in the prompt.")
        if len(label_list) == 0:
            raise ValueError("label_list is missing in the prompt.")

        # ---- 调用 SSR 模板函数重建为标准 prompt ----
        # construct_prompt_chosen_reasoning 会根据任务类型和提取的图信息，
        # 生成带有 <Sampled_subgraph>、推理、<Answer> 等结构的标准 SSR prompt
        prompt = construct_prompt_chosen_reasoning(task_type, target_nodes, node_information, edge_information, label_list, group_size)

        # ---- 按 (数据集+任务类型) 组织数据 ----
        # task_key 格式如 "cora_node+nc" 或 "fb15k237+lp"
        task_key = f"{dataset}+{task_type}"
        if task_key not in gofa_test_data:
            # 首次遇到此 task_key，初始化数据结构
            gofa_test_data[task_key] = {
                "messages": [[{"role": "user", "content": prompt}]],
                "truth_labels": [truth_label]
            }
        else:
            # 追加到已有 task_key 的列表中
            gofa_test_data[task_key]["messages"].append([{"role": "user", "content": prompt}])
            gofa_test_data[task_key]["truth_labels"].append(truth_label)

    # ---- 为每个 task_key 构建答案集合（去重后的所有标准答案） ----
    for task_key in gofa_test_data:
        gofa_test_data[task_key]["answer_set"] = set(gofa_test_data[task_key]["truth_labels"])

    return gofa_test_data


# =============================================================================
# get_answer_set() — 获取数据集的合法答案集合
# =============================================================================
# 参数：
#   dataset : 数据集名称（如 "cora_node", "wikics", "fb15k237"）
#
# 返回值：
#   该数据集所有合法答案的小写集合（set），用于后续判断模型输出是否是"已知答案"
#
# 为什么需要这个函数：
#   在评估时，模型的输出可能不属于任何已知的标签（即模型"编造"了一个答案）。
#   本函数返回的 answer_set 是判断基准——模型输出如果在集合内则是 valid answer，
#   否则是 unknown answer（未知/编造的回答）。
#
# 答案集合的三种来源：
#   1. 硬编码（answer_set_dic）：对于 cora_node, wikics, products 等数据集，
#      标签集合是固定的已知集合，直接写在代码中
#   2. 从 .pt 文件读取（fb15k237, wikics, products）：
#      部分数据集除了硬编码的标签外，还需要从 graph_bert_encoded.pt 文件中
#      读取额外的 label_list 以扩充答案集合（取两个来源的并集）
#   3. few-shot 变体复用：
#      cora_node_2ways / products_10ways / products_5ways / wikics_5ways
#      这些是原始数据集的 few-shot 变体，标签集合与原始数据集完全相同
# =============================================================================
def get_answer_set(dataset):
    # 硬编码的答案集合（这些数据集的标签是固定的，不会变化）
    answer_set_dic = {
        # cora_node：论文主题分类（7 个 AI 子领域）
        "cora_node": set(['rule learning', 'genetic algorithms', 'case-based', 'theory', 'neural networks', 'probabilistic methods', 'reinforcement learning']),
        # wikics：维基百科计算机科学文章分类（10 个 CS 子领域）
        "wikics": set(['operating systems', 'computer security', 'programming language topics', 'computational linguistics', 'distributed computing architecture', 'computer architecture', 'internet protocols', 'databases', 'web technology', 'computer file systems']),
        # products：Amazon 商品类别分类（47 个商品类别）
        "products": set(['baby', 'camera & photo', 'magazine subscriptions', 'cds & vinyl', 'kindle store', 'computers', 'tools & home improvement', 'software', 'video games', 'books', 'musical instruments', 'movies & tv', 'label 25', 'kitchen & dining', 'toys & games', 'baby products', 'gps & navigation', 'gift cards', 'digital music', 'grocery & gourmet food', 'home improvement', 'furniture & decoration', 'health & personal care', 'all electronics', 'electronics', 'all beauty', '#508510', 'beauty', 'mp3 players & accessories', 'home & kitchen', 'collectibles & fine art', 'purchase circles', 'car electronics', 'office products', 'patio, lawn & garden', 'pet supplies', 'clothing, shoes & jewelry', 'arts, crafts & sewing', 'industrial & scientific', 'buy a kindle', 'amazon fashion', 'luxury beauty', 'sports & outdoors', 'cell phones & accessories', 'office & school supplies', 'automotive', 'appliances'])
    }
    # few-shot 变体复用原始数据集的标签集合
    answer_set_dic["cora_node_2ways"] = answer_set_dic["cora_node"]
    answer_set_dic["products_10ways"] = answer_set_dic["products"]
    answer_set_dic["products_5ways"] = answer_set_dic["products"]
    answer_set_dic["wikics_5ways"] = answer_set_dic["wikics"]

    # ---- 根据数据集名称，采用不同策略获取答案集合 ----
    if dataset in ["fb15k237"]:
        # fb15k237：从 .pt 文件中读取 label_list
        data_file = "../data/test_data/raw_data/{}/{}_graph_bert_encoded.pt".format(dataset, dataset)
        data = torch.load(data_file, weights_only=False)
        answer_set = set([label.lower().strip() for label in data.label_list])

    elif dataset in ["cora_node", "cora_node_2ways", "products_10ways", "products_5ways", "wikics_5ways"]:
        # few-shot 变体和 cora_node：仅使用硬编码的答案集合
        answer_set = answer_set_dic[dataset]

    elif dataset in ["wikics", "products"]:
        # wikics 和 products：取 .pt 文件中的标签 与 硬编码标签的并集
        # 这样既能覆盖数据集中的完整标签，又能兜底 .pt 文件可能不完整的情况
        data_file = "../data/test_data/raw_data/{}/{}_graph_bert_encoded.pt".format(dataset, dataset)
        data = torch.load(data_file, weights_only=False)
        answer_set = set([label.lower().strip() for label in data.label_list])
        answer_set = answer_set | answer_set_dic[dataset]

    else:
        # 不支持的数据集
        raise ValueError("Unsupported dataset")

    return answer_set


# =============================================================================
# get_answer() — 从模型原始响应中提取最终答案
# =============================================================================
# 参数：
#   response : 模型的完整响应字符串（包含推理过程）
#
# 返回值：
#   (ans_type, ans) 元组：
#     ans_type : "normal"（成功提取）或 "wrong format"（格式错误）
#     ans      : 提取出的答案文本（格式错误时为空字符串）
#
# 提取逻辑：
#   模型响应遵循 DeepSeek-R1 风格的格式：
#     <think>推理过程...</think>
#     <Chosen_subgraph_reasoning>
#       ...
#       <Answer>最终答案</Answer>
#     </Chosen_subgraph_reasoning>
#
#   提取步骤：
#     1. 如果响应包含 </think>，则从 </think> 之后开始解析
#     2. 用正则提取 <Chosen_subgraph_reasoning> 标签内的内容
#     3. 在该内容中提取 <Answer> 标签内的文本
#     4. 如果任何步骤失败，返回 "wrong format"
# =============================================================================
def get_answer(response):
    try:
        # 检查响应是否包含 </think> 结束标签
        if "</think>" in response:
            # 分割响应：只保留 </think> 之后的部分（丢弃模型内部推理）
            response = response.split("</think>", 1)[-1]

            # 提取 <Chosen_subgraph_reasoning>...</Chosen_subgraph_reasoning> 内容
            Chosen_subgraph_reasoning_pattern = r"<Chosen_subgraph_reasoning>(.*?)</Chosen_subgraph_reasoning>"
            Chosen_subgraph_reasoning_match = re.findall(Chosen_subgraph_reasoning_pattern, response, re.DOTALL)
            Chosen_subgraph_reasoning = Chosen_subgraph_reasoning_match[-1].strip()

            # 从 Chosen_subgraph_reasoning 中提取 <Answer>...</Answer> 内容
            pattern = r'<Answer>(.*?)</Answer>'
            match = re.findall(pattern, Chosen_subgraph_reasoning, re.DOTALL)
            ans = match[-1].strip()  # 取最后一个匹配（避免嵌套标签干扰）
            ans_type = "normal"
            return ans_type, ans
        else:
            # 响应不包含 </think> 标签 → 格式不符合预期
            ans_type = "wrong format"
            ans = ""
            return ans_type, ans
    except Exception as e:
        # 任何解析异常（如正则匹配失败、索引越界等）都视为格式错误
        ans_type = "wrong format"
        ans = ""
        return ans_type, ans


# =============================================================================
# check_answer() — 判断模型答案的正确性
# =============================================================================
# 参数：
#   ans_type    : get_answer() 返回的答案类型（"normal" 或 "wrong format"）
#   ans         : get_answer() 返回的答案文本
#   truth_label : 标准答案（ground truth）
#   task_type   : 任务类型
#   answer_set  : 合法答案集合（由 get_answer_set() 返回）
#
# 返回值：
#   一个字典，包含：
#     - ans_type    : 答案分类结果（见下方四种类型）
#     - truth_label : 标准答案
#     - ans         : 模型给出的答案
#
# 答案分类逻辑（四种类型）：
#   - "wrong format"  : 模型响应格式错误，无法提取答案
#   - "correct answer": 答案在合法集合中 且 等于标准答案 → 正确
#   - "wrong answer"  : 答案在合法集合中 但不等于标准答案 → 错误但格式合规
#   - "unknown answer": 答案不在合法集合中 → 模型可能"编造"了一个不存在的标签
#
# 特殊情况：如果 ans 在 truth_label 中（部分匹配），也视为正确。
#   例如模型回答 "rule learning" 而标准答案也是 "rule learning"，直接匹配。
# =============================================================================
def check_answer(ans_type, ans, truth_label, task_type, answer_set):
    if ans_type == "wrong format":
        # 格式错误：无法提取有效答案
        ans_dic = {
            "ans_type": ans_type,
            "truth_label": truth_label,
            "ans": ans
        }
    else:
        # 将答案和标准答案都转为小写进行比较（不区分大小写）
        ans = ans.lower().strip()
        if ans in answer_set:
            # 答案在合法集合中
            if ans == truth_label:
                ans_type = "correct answer"    # 正确回答
            else:
                ans_type = "wrong answer"      # 错误但合法的回答
        else:
            # 答案不在合法集合中 → 可能是未知/编造的回答
            if ans in truth_label:
                # 虽然不在 answer_set 中，但恰好是标准答案的子串 → 仍算正确
                ans_type = "correct answer"
            else:
                ans_type = "unknown answer"    # 未知回答（模型输出了无效标签）
        ans_dic = {
            "ans_type": ans_type,
            "truth_label": truth_label,
            "ans": ans
        }
    return ans_dic


# =============================================================================
# test_answer() — 批量评估模型响应
# =============================================================================
# 参数：
#   json_list  : 模型响应列表，每项包含 {"response", "index", "truth_label"}
#   task_type  : 任务类型
#   answer_set : 合法答案集合
#
# 返回值：
#   评估结果列表，每项是一个由 check_answer() 返回的字典（附加了 index 字段）
# =============================================================================
def test_answer(json_list, task_type, answer_set):
    answers = []
    for item in json_list:
        response = item["response"]           # 模型原始响应
        index = item["index"]                 # 样本编号
        truth_label = item["truth_label"].lower().strip()  # 标准答案（统一小写）
        # 提取答案 → 判断正确性
        ans_type, ans = get_answer(response)
        ans_dic = check_answer(ans_type, ans, truth_label, task_type, answer_set)
        ans_dic["index"] = index
        answers.append(ans_dic)
    return answers


# =============================================================================
# evaluate_answer() — 对单个 (数据集, 任务类型) 组合执行完整评估
# =============================================================================
# 参数：
#   dataset   : 数据集名称
#   task_type : 任务类型
#   json_list : 模型响应列表
#
# 返回值：
#   评估后的答案列表（由 test_answer() 返回）
# =============================================================================
def evaluate_answer(dataset, task_type, json_list):
    # 先获取该数据集的合法答案集合
    answer_set = get_answer_set(dataset)
    # 然后批量评估所有回答
    evaluated_answers = test_answer(json_list, task_type, answer_set)

    return evaluated_answers


# =============================================================================
# cal_metrics() — 计算评估指标
# =============================================================================
# 参数：
#   answers : evaluate_answer() 返回的评估结果列表
#
# 返回值：
#   一个包含以下字段的字典：
#     - total         : 总样本数
#     - correct       : 回答正确的样本数
#     - wrong_answer  : 回答错误的样本数（答案合法但不等于标准答案）
#     - unknown_answer: 未知回答的样本数（答案不在合法集合中）
#     - wrong_format  : 格式错误的样本数（无法提取答案）
#     - accuracy      : 整体准确率（correct / total）
#     - label_metrics : 按标签细分的指标（每个标签的准确率等）
#
# 计算逻辑：
#   1. 遍历所有答案，分别统计四种类型的数量
#   2. 按 truth_label 分组，计算每个标签类别下的准确率
#   3. 汇总为整体准确率
# =============================================================================
def cal_metrics(answers):
    total = len(answers)
    # 统计四种答案类型的数量
    correct_answer = sum(1 for ans in answers if ans["ans_type"] == "correct answer")
    wrong_answer = sum(1 for ans in answers if ans["ans_type"] == "wrong answer")
    unknown_answer = sum(1 for ans in answers if ans["ans_type"] == "unknown answer")
    wrong_format = sum(1 for ans in answers if ans["ans_type"] == "wrong format")

    # ---- 按标签类别计算细粒度指标 ----
    # 这可以揭示模型是否在某些特定类别上表现特别差（例如总把某个标签预测为另一个）
    label_metrics = {}
    for ans in answers:
        ans_type = ans["ans_type"]
        truth_label = ans["truth_label"]
        # 初始化标签条目
        if truth_label not in label_metrics:
            label_metrics[truth_label] = {
                "total": 0,
                "correct": 0,
                "wrong_answer": 0,
                "unknown_answer": 0,
                "wrong_format": 0
            }

        # 累加该标签的各类计数
        label_metrics[truth_label]["total"] += 1
        if ans_type == "correct answer":
            label_metrics[truth_label]["correct"] += 1
        elif ans_type == "wrong answer":
            label_metrics[truth_label]["wrong_answer"] += 1
        elif ans_type == "unknown answer":
            label_metrics[truth_label]["unknown_answer"] += 1
        elif ans_type == "wrong format":
            label_metrics[truth_label]["wrong_format"] += 1

    # 计算每个标签的准确率
    for label in label_metrics:
        label_total = label_metrics[label]["total"]
        label_correct = label_metrics[label]["correct"]
        label_accuracy = label_correct / label_total if label_total > 0 else 0.0
        label_metrics[label]["accuracy"] = label_accuracy

    # 计算整体准确率
    accuracy = correct_answer / total if total > 0 else 0.0

    # 汇总指标
    metrics = {
        "total": total,
        "correct": correct_answer,
        "wrong_answer": wrong_answer,
        "unknown_answer": unknown_answer,
        "wrong_format": wrong_format,
        "accuracy": accuracy,
        "label_metrics": label_metrics
    }

    return metrics


# =============================================================================
# main() — 评测主函数
# =============================================================================
# 参数：
#   data_path     : GOFA 测试 CSV 文件路径
#   save_dir      : 结果保存目录
#   seed          : 随机种子
#   temperature   : 推理温度
#   choose_server : 使用的评测服务器名称列表
#   group_size    : 子图采样分组大小
#
# 执行流程：
#   1. 加载 GOFA 测试数据并重建 SSR prompt
#   2. 遍历每个 (数据集, 任务类型) 组合
#   3. 如果推理结果不存在，调用 eval_api 执行推理并保存
#   4. 加载已有的推理结果，评估答案正确性
#   5. 计算评估指标并保存到 JSON 文件
# =============================================================================
def main(data_path, save_dir, seed, temperature, choose_server, group_size):

    # ---- 步骤 1：加载测试数据并重建为 SSR prompt ----
    print(f"Loading test data from {data_path}...")
    test_data = get_gofa_prompt(data_path, group_size)

    # 推理参数配置
    sampling_params = {
        "temperature": temperature,   # 温度控制随机性
        "max_tokens": 16384,          # 最大生成 token 数
        "top_k": 50,                  # Top-K 采样
        "seed": seed                  # 随机种子（保证可复现性）
    }

    # ---- 步骤 2：遍历每个任务组合执行推理和评估 ----
    for task_key in test_data:
        print(f"Processing task: {task_key}...")
        print(f"Number of samples: {len(test_data[task_key]['messages'])}")

        # 解析任务标识符（如 "cora_node+nc" → dataset="cora_node", task_type="nc"）
        dataset, task_type = task_key.split('+')
        # 推理结果和评估指标的保存路径
        save_path = f'{save_dir}/{dataset}/{dataset}_{task_type}_inference_result.jsonl'
        metrics_save_path = f'{save_dir}/{dataset}/{dataset}_{task_type}_metrics_result.json'

        # ---- 步骤 3：执行模型推理（如果结果文件还不存在） ----
        if not os.path.exists(save_path):
            messages = test_data[task_key]["messages"]
            truth_labels = test_data[task_key]["truth_labels"]
            # 确保保存目录存在
            if os.path.exists(f'{save_dir}/{dataset}') is False:
                os.makedirs(f'{save_dir}/{dataset}')

            # 调用推理 API，将结果保存到文件
            eval_api(save_path, messages, truth_labels, sampling_params, choose_server)

        # ---- 步骤 4：加载已有推理结果，执行评估 ----
        print(f"Loading existing inference results from {save_path}...")
        json_list = []
        with open(save_path, 'r') as f:
            for line in f:
                json_list.append(json.loads(line.strip()))

        # 评估答案正确性并计算指标
        evaluated_answers = evaluate_answer(dataset, task_type, json_list)
        metrics = cal_metrics(evaluated_answers)
        print(f"Acc for {task_key}: {metrics['accuracy']*100:.4f}%")

        # 保存评估指标 JSON
        with open(metrics_save_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # 保存逐条答案评估结果（方便后续 case study 和错误分析）
        with open(f'{save_dir}/{dataset}/{dataset}_{task_type}_evaluated_answers.jsonl', 'w') as f:
            for item in evaluated_answers:
                f.write(json.dumps(item) + '\n')


# =============================================================================
# eval_api() — 调用推理 API 并支持自动重试
# =============================================================================
# 参数：
#   save_path       : 最终推理结果保存路径
#   messages        : prompt 消息列表
#   truth_labels    : 对应的标准答案列表
#   sampling_params : 推理采样参数
#   choose_server   : 使用的服务器名称列表
#
# 重试机制：
#   模型推理结果中可能存在格式不合规的样本（无法提取 <Answer>）。
#   本函数采用循环重试策略：
#     1. 首次调用 reason() 对所有未生成的样本执行推理
#     2. 检查格式合规率：如果错误样本数 > 总数/1000（即错误率 > 0.1%），
#        剔除格式错误样本，保留合规样本，重新推理剩余样本
#     3. 最多重试 8 次（避免无限循环）
#
# 为什么需要重试：
#   推理过程中的随机采样可能导致某些 prompt 生成的响应格式不正确。
#   通过多次重试（每次使用不同的随机种子/temperature），可以提高最终
#   输出中格式合规样本的比例，确保评估结果可靠。
# =============================================================================
def eval_api(save_path, messages, truth_labels, sampling_params, choose_server):
    # 导入共享的批量推理工具函数
    from utils.utils import reason

    # 构建样本列表：每个样本包含 prompt 和 truth_label
    samples = []
    for i in range(len(messages)):
        row = {
            "index": i,
            "prompt": messages[i][0]['content'],
            "truth_label": truth_labels[i]
        }
        samples.append(row)

    # 获取所选服务器的 URL 列表
    choose_name = choose_server
    url_list = [URL_LIST_EVAL[name] for name in choose_name]

    # ---- 重试循环 ----
    trying_times = 0
    while True:
        print(f"Reasoning attempt number: {trying_times + 1}")

        # 中间推理结果保存路径（与最终结果区分，方便重试时追加重试结果）
        save_path_reasoning = save_path.replace("_inference_result.jsonl", "_reasoning_result.jsonl")

        # 加载已经生成过的样本索引（如果文件已存在），避免重复推理
        generated_sample_indicies = set()
        if os.path.exists(save_path_reasoning):
            print(f"Loading generated samples from {save_path_reasoning}")
            with open(save_path_reasoning, encoding='utf-8') as f:
                for raw_line in f:
                    sample = json.loads(raw_line)
                    generated_sample_indicies.add(int(sample["index"]))

        print(f"{len(generated_sample_indicies)} generated samples")

        # 筛选出尚未生成推理结果的样本
        targets = [i for i in range(len(samples)) if i not in generated_sample_indicies]

        # 调用批量推理函数（通过 HTTP API 并发发送请求）
        reason(samples, targets, save_path_reasoning, url_list, sampling_params, print_info=True)

        # ---- 格式检查：过滤掉格式不合规的样本 ----
        lines_new = []
        with open(save_path_reasoning, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line)
                ans_type, ans = get_answer(line["response"])
                if ans_type == "normal":
                    # 只保留能成功提取答案的样本
                    lines_new.append(line)

        # ---- 判断是否满足退出条件 ----
        # 条件 1：格式错误样本数 ≤ 总样本数的 1/1000（即错误率 < 0.1%）
        # 条件 2：重试次数已达上限（8 次）
        if len(samples) - len(lines_new) <= len(samples) // 1000 or trying_times >= 8:
            break

        # 保存当前合规的样本，下一轮只重试不合规的（通过 targets 过滤实现）
        with open(save_path_reasoning, 'w', encoding='utf-8') as f:
            for line in lines_new:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        trying_times += 1

    # ---- 最终整理：按 index 排序并保存为最终推理结果 ----
    with open(save_path_reasoning, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f.readlines()]

    # 按样本索引排序，确保输出结果与输入顺序一致
    lines = sorted(lines, key=lambda x: x['index'])
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


# =============================================================================
# 脚本入口：解析命令行参数并调用 main()
# =============================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GraphSSR 模型评测脚本 —— 在 GOFA 基准上评估训练好的图推理模型")

    # 测试数据路径（GOFA CSV 文件）
    parser.add_argument('-d', "--data_path", type=str, help="GOFA 测试集的 CSV 文件路径")
    # 结果保存目录
    parser.add_argument('-s', "--save_dir", type=str, help="评估结果的保存目录")
    # 随机种子（默认 42）
    parser.add_argument('-e', "--seed", type=int, default=42, help="随机种子，用于控制推理采样的可复现性")
    # 推理温度（默认 0.5，适中的随机性）
    parser.add_argument('-t', "--temperature", type=float, default=0.5, help="推理时的 temperature 参数")
    # 评测服务器列表（对应 URL_LIST_EVAL 中的 key）
    parser.add_argument('-c', "--choose_server", nargs='+', help='使用的评测服务器名称列表，如 eval_server_0')
    # 子图采样分组大小（默认 5）
    parser.add_argument('-g', "--group_size", type=int, default=5, help="子图采样时每个分组的节点数量")

    args = parser.parse_args()
    data_path = args.data_path
    save_dir = args.save_dir
    seed = args.seed
    temperature = args.temperature
    choose_server = args.choose_server
    main(data_path, save_dir, seed, temperature, choose_server, args.group_size)
