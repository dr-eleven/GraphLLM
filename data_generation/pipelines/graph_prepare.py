"""
GraphSSR 图数据预处理模块。

本文件实现的功能：
- 从 GraphR1 的 SFT 和 RL 数据记录中提取图结构字段（目标节点、邻居节点、边关系、标签等）。
- 加载节点摘要和采样子图，将其转换为可供 prompt 构造使用的标准化格式。
- 维护每个数据集的任务采样目标配额，供 RL 数据生成阶段使用。

在训练流程中的角色：
- SFT 阶段：将原始 GraphR1 SFT CSV 数据转换为标准化的图元数据。
- RL 数据构建阶段：将采样子图和标签输入到 prompt 生成环节，为 PPO 训练做准备。
"""

import re
import os
import sys
import json
import random
import torch
import pandas as pd
from tqdm import tqdm

root_dir = "../.."
sys.path.append(os.path.abspath(root_dir))

# 各数据集的采样配额配置
# 每个 (数据集, 任务类型) 组合指定 easy / medium / hard 三个难度级别需要的样本数
# nc=节点分类, lp=链接预测, lc=链接分类
dataset_stat = {
    ("arxiv", "nc"): {"easy": 200, "medium": 200, "hard": 100},
    ("arxiv", "lp"): {"easy": 200, "medium": 200, "hard": 100},
    ("children", "nc"): {"easy": 200, "medium": 200, "hard": 100},
    ("children", "lp"): {"easy": 200, "medium": 200, "hard": 100},
    ("computer", "nc"): {"easy": 200, "medium": 200, "hard": 100},
    ("computer", "lp"): {"easy": 200, "medium": 200, "hard": 100},
    ("photo", "nc"): {"easy": 200, "medium": 200, "hard": 100},
    ("photo", "lp"): {"easy": 200, "medium": 200, "hard": 100},
    ("pubmed", "nc"): {"easy": 200, "medium": 200, "hard": 100},
    ("pubmed", "lp"): {"easy": 200, "medium": 200, "hard": 100},
    ("sports", "nc"): {"easy": 200, "medium": 200, "hard": 100},
    ("sports", "lp"): {"easy": 200, "medium": 200, "hard": 100},
    ("reddit", "nc"): {"easy": 200, "medium": 200, "hard": 100},
    ("reddit", "lp"): {"easy": 200, "medium": 200, "hard": 100},
    ("history", "nc"): {"easy": 200, "medium": 200, "hard": 100},
    ("history", "lp"): {"easy": 200, "medium": 200, "hard": 100},
    ("wn18rr", "lc"): {"easy": 800, "medium": 800, "hard": 400},
}

# 链接预测类型映射：将数据集名称映射为自然语言的关系类型描述
link_prediction_type_map = {
    "arxiv": "citation",
    "pubmed": "citation",
    "children": "co-purchased or co-viewed",
    "history": "co-purchased or co-viewed",
    "computer": "co-purchased or co-viewed",
    "photo": "co-purchased or co-viewed",
    "sports": "co-purchased or co-viewed",
    "instagram": "following",
    "reddit": "replying"
}

# 从 GraphR1 SFT 数据集的一行中提取图结构信息
# 使用正则表达式解析 prompt 和 response，提取目标节点、邻居节点信息、
# 边信息、候选标签列表和正确答案标签
def extract_info_from_sft_data(row):
    prompt = row['prompt']
    response = row['response']
    dataset = row['dataset']
    task = row['task']

    # 用正则提取第一行中的目标节点 ID（如 node0, node1）
    target_node_pattern = r'node\d+'
    target_node_matches = re.findall(target_node_pattern, prompt.split("\n")[0], re.DOTALL)
    target_nodes = [match.strip() for match in target_node_matches]

    # 提取所有节点的文本描述（如 "node0: This paper is about..."）
    node_pattern = r'node\d+:.*?\n'
    node_matches = re.findall(node_pattern, prompt, re.DOTALL)
    node_information = [match.strip() for match in node_matches]

    # 提取边关系（如 <node0, node1>）
    edge_pattern = r'<node\d+, node\d+>'
    edge_matches = re.findall(edge_pattern, prompt, re.DOTALL)
    edge_information = [match.strip() for match in edge_matches]

    if task == 'nc' or task == 'lc':
        label_pattern = r'Select strictly from:(.*?)\. Respond only with the'
        matches = re.findall(label_pattern, prompt, re.DOTALL)
        if dataset == "instagram":
            label_list = [label.strip().strip('"') for label in matches[0].split('", "')]
        else:
            label_list = [label.strip().strip("'") for label in matches[0].split("', '")]
    elif task == 'lp':
        label_pattern = r'select strictly from: (.*?)\.Respond only with the'
        matches = re.findall(label_pattern, prompt, re.DOTALL)
        label_list = [label.strip().strip("'") for label in matches[0].split("' or '")]
    else:
        label_list = []

    # 从 response 的 <｜end▁of▁thinking｜> 的 </think> 之后提取最终答案
    split_response = response.split("</think>", 1)[-1]
    pattern = r'Answer:\s*(.*?)\s*\n'
    match = re.findall(pattern, split_response, re.IGNORECASE | re.DOTALL)
    if len(match) == 0:
        match = re.findall(r'Answer:\s*(.+)', split_response, re.IGNORECASE | re.DOTALL)
    truth_label = match[-1].strip()

    return task, dataset, target_nodes, node_information, edge_information, label_list, truth_label

# 从 GraphR1 RL 测试数据集的一行中提取图结构信息
# 解析 parquet 格式数据中的 prompt，提取目标节点、邻居节点、边关系、候选标签、
# ground truth 等字段，并检查 prompt 是否缺少必要的组成部分（节点或标签）
def extract_info_from_rl_data(row):
    prompt = row['prompt'][0]["content"]
    dataset = row['dataset']
    task = row['task']
    ground_truth = row['ground_truth']
    data_source = "graphr2_subgraph"
    reward_model = row['reward_model']
    extra_info = row['extra_info']

    target_node_pattern = r'node\d+'
    target_node_matches = re.findall(target_node_pattern, prompt.split("\n")[0], re.DOTALL)
    target_nodes = [match.strip() for match in target_node_matches]

    node_pattern = r'node\d+:.*?\n'
    node_matches = re.findall(node_pattern, prompt, re.DOTALL)
    node_information = [match.strip() for match in node_matches]

    edge_pattern = r'<node\d+, node\d+>'
    edge_matches = re.findall(edge_pattern, prompt, re.DOTALL)
    edge_information = [match.strip() for match in edge_matches]

    if task == 'nc':
        label_pattern = r'Select strictly from:(.*?)\.'
        matches = re.findall(label_pattern, prompt, re.DOTALL)
        label_list = [label.strip().strip("'") for label in matches[0].split("', '")]
        if dataset == "cora_node":
            label_list = [label.replace("Theroy", "Theory") for label in label_list]
    elif task == 'lc':
        label_pattern = r'Choose from the following:(.*?)\.'
        matches = re.findall(label_pattern, prompt, re.DOTALL)
        if len(matches) == 0:
            label_pattern = r'Select strictly from:(.*?)\.'
            matches = re.findall(label_pattern, prompt, re.DOTALL)
        label_list = [label.strip().strip("'") for label in matches[0].split(';')]
    elif task == 'lp':
        label_pattern = r'select strictly from: (.*?)\.Respond only with the'
        matches = re.findall(label_pattern, prompt, re.DOTALL)
        label_list = [label.strip().strip("'") for label in matches[0].split("' or '")]
    else:
        label_list = []

    graph_info = {
        "dataset": dataset,
        "task": task,
        "target_nodes": target_nodes,
        "node_information": node_information,
        "edge_information": edge_information,
        "label_list": label_list,
        "ground_truth": ground_truth,
        "reward_model": reward_model,
        "extra_info": extra_info
    }

    graph_info = graph_info_format(graph_info)

    skip_status = False
    if len(target_node_matches) == 0:
        raise ValueError("target_nodes is missing in the prompt")
    if len(node_matches) == 0:
        raise ValueError("node_matches is missing in the prompt")
    if len(node_matches) >= 2:
        if len(edge_matches) == 0:
            skip_status = True
    if len(label_list) == 0:
        raise ValueError("label_list is missing in the prompt")

    return graph_info, skip_status

# 将图信息格式化为 prompt 可直接使用的字符串形式
# - 目标节点用逗号连接
# - 节点信息按行排列（每行缩进两空格）
# - 边信息用逗号连接
# - 标签列表用 < > 包裹
def graph_info_format(graph_info):

    graph_info['target_nodes'] = ', '.join(graph_info['target_nodes'])
    graph_info['node_information'] = '\n'.join(["  " + node_information for node_information in graph_info['node_information']])
    graph_info['edge_information'] = ', '.join(graph_info['edge_information'])
    graph_info["label_list"] = "<" + ">, <".join(graph_info["label_list"]) + ">"

    return graph_info

# 从 GraphR1 SFT 数据集加载并提取图信息列表
# 首次调用时从原始 CSV 中提取信息并缓存为 JSONL 文件，后续直接从缓存加载
# 返回格式化后的图信息列表，供 SFT prompt 构造使用
def get_graph_from_graphr1_sft(root_dir):
    
    graph_info_data_path = f"{root_dir}/data/train_data/raw_data/graphr1/graphr1_sft_graph_info.jsonl"
    
    if not os.path.exists(graph_info_data_path):
        print("Generating graphr1 sft graph info data for the first time...")

        # 加载原始 SFT 数据
        raw_data_path = f"{root_dir}/data/train_data/raw_data/graphr1/sft_train_data_filter2048_10000_balance.csv"
        raw_sft_data = pd.read_csv(raw_data_path)

        # 遍历每一行提取图信息并写入 JSONL 文件
        f = open(graph_info_data_path, 'w', encoding='utf-8')
        for index, row in tqdm(raw_sft_data.iterrows(), total=len(raw_sft_data)):
            if row['task'] not in ["nc", "lp", "lc"]:
                continue
            task, dataset, target_nodes, node_information, edge_information, label_list, truth_label = extract_info_from_sft_data(row)
            graph_info = {
                "dataset": dataset,
                "task": task,
                "target_nodes": target_nodes,
                "node_information": node_information,
                "edge_information": edge_information,
                "label_list": label_list,
                "truth_label": truth_label
            }
            f.write(json.dumps(graph_info) + "\n")
        f.close()

    # 从缓存文件中加载并格式化图信息
    print("Loading graphr1 sft graph info data...")
    graph_info_list = []
    with open(graph_info_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            graph_info = json.loads(line.strip())
            graph_info = graph_info_format(graph_info)
            graph_info_list.append(graph_info)

    return graph_info_list

# 为 RL 数据集任务获取新的训练样本
# 从预处理的子图池中随机采样未使用过的子图，结合节点摘要信息构造格式化的图信息列表
# 返回 (已采样的索引列表, 图信息列表)
def get_data_for_rl_dataset_task(dataset, task, subgraph_data_path, summary_data_path, sample_num, have_sampled_index):

    # 加载预处理好的子图数据（由 preprocess.py 生成）
    print("Loading subgraph data.")
    subgraph_data = torch.load(subgraph_data_path, weights_only=False)
    print("Finished loading subgraph data.")

    # 检查是否所有子图已被采样过
    if len(have_sampled_index) >= len(subgraph_data):
        print("All subgraphs have been sampled for dataset {}, task {}".format(dataset, task))
        return []

    # 从未采样过的子图中随机选择 sample_num 个新样本
    all_sample_indices = list(range(len(subgraph_data)))
    waiting_indices = list(set(all_sample_indices) - set(have_sampled_index))
    
    sample_index = random.sample(waiting_indices, min(sample_num, len(waiting_indices)))
    sample_index = sorted(sample_index)
    subgraph_data_sample = [(i, subgraph_data[i]) for i in sample_index]
    
    print("All {} subgraphs, have sampled {} subgraphs, this time sample {} subgraphs:".format(len(subgraph_data), len(have_sampled_index), len(subgraph_data_sample)))

    # 加载节点摘要数据（包含每个节点的文本描述）
    print("Loading summary data.")
    summary_data = []
    with open(summary_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            summary_data.append(json.loads(line.strip()))
    print("Finished loading summary data.")

    # 构造图信息列表，为每个采样子图生成标准化的图结构描述
    graph_info_list = []
    for si, sample in subgraph_data_sample:
        # 构建目标节点、节点文本描述和边关系的列表
        target_nodes = [f"node{idx}" for idx in sample.target_node.tolist()]
        node_information = [f"node{idx}:{summary_data[info_idx]['response']}" for idx, info_idx in enumerate(sample.n_id.tolist())]
        edge_information = [f"<node{src}, node{dst}>" for src, dst in zip(sample.edge_index[0].tolist(), sample.edge_index[1].tolist())]
        if task == 'nc':
            label_list = sample.label_list
            truth_label = sample.label_list[sample.y[0].item()]
        elif task == 'lp':
            truth_label = "Yes, they have {} relationships".format(link_prediction_type_map[dataset]) if sample.edge_label[0].item() == 1 else "No, they do not have {} relationships".format(link_prediction_type_map[dataset])
            label_list = ["Yes, they have {} relationships".format(link_prediction_type_map[dataset]), "No, they do not have {} relationships".format(link_prediction_type_map[dataset])]
        elif task == 'lc':
            label_list = sample.label_list
            truth_label = sample.label_list[sample.edge_label[0].item()]
        else:
            truth_label = ""
            label_list = []

        # 格式化图信息为 prompt 就绪的字符串格式
        graph_info = {
            "dataset": dataset,
            "task": task,
            "sample_index": si,
            "target_nodes": target_nodes,
            "node_information": node_information,
            "edge_information": edge_information,
            "label_list": label_list,
            "truth_label": truth_label
        }

        graph_info = graph_info_format(graph_info)

        graph_info_list.append(graph_info)

    return sample_index, graph_info_list
    
# 为 RL 数据生成阶段准备所有数据集的图信息
# 遍历 dataset_stat 中定义的 (数据集, 任务) 组合：
# - 如果某个组合的 easy/medium/hard 样本已收集足够，则跳过
# - 否则计算还需多少样本，从子图池中采样新数据
# 返回按 (dataset, task) 键组织的图信息字典
def generate_data_for_rl(have_sampled_index_dic, checked_sample_dic):
    graph_info_dic = {}
    for key in dataset_stat.keys():
        dataset, task = key
        print(dataset, task)

        raw_data_dir = f"{root_dir}/data/train_data/raw_data/{dataset}"
        processed_data_dir = f"{root_dir}/data/train_data/processed_data/{dataset}"

        summary_data_path = f"{raw_data_dir}/{dataset}_summary.jsonl"
        subgraph_data_path = f"{processed_data_dir}/{dataset}_{task}_subgraph_bert_encoded.pt"

        # 检查当前 (dataset, task) 是否已收集足够的 easy/medium/hard 样本
        if len(checked_sample_dic[key]['easy']) >= dataset_stat[key]['easy'] and \
            len(checked_sample_dic[key]['medium']) >= dataset_stat[key]['medium'] and \
            len(checked_sample_dic[key]['hard']) >= dataset_stat[key]['hard']:
            print("Enough samples for dataset {}, task {}, skip.".format(dataset, task))
            print([len(checked_sample_dic[key][difficulty]) for difficulty in checked_sample_dic[key].keys()])
            graph_info_dic[key] = ([], [])
        # 如果还需要更多样本，计算需采样数（至少 500，或 needed_num*2）并调用采样函数
        else:
            needed_num = sum([dataset_stat[key][difficulty] - len(checked_sample_dic[key][difficulty]) for difficulty in checked_sample_dic[key].keys()])
            print("Need to sample {} more samples.".format(needed_num))
            print([len(checked_sample_dic[key][difficulty]) for difficulty in checked_sample_dic[key].keys()])
            # 调用采样函数获取新的子图数据（采样数量取 needed_num*2 和 500 的较大值）
            graph_info_dic[key] = get_data_for_rl_dataset_task(dataset, task, subgraph_data_path, summary_data_path, max(needed_num * 2, 500), have_sampled_index_dic[key])

    return graph_info_dic
