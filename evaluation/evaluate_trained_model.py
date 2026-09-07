"""
GraphSSR 的 GOFA 风格基准评估入口模块。

本文件实现的功能：
- 将 GOFA 测试集的原始 prompt 重建为 GraphSSR 期望的 SSR prompt 格式。
- 将重构后的 prompt 发送到部署好的推理服务器。
- 解析模型响应并按数据集和任务类型计算准确率和格式合规率等指标。

在训练流程中的角色：
- 不属于参数更新阶段。
- 训练后评估阶段，用于评估最终 SFT 或 RL 检查点的性能。
"""

import re
import os
import sys
import json
import subprocess
from pathlib import Path
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# 定义各数据集对应的链接预测关系类型映射
link_prediction_type_map = {
    "arxiv": "citation",
    "pubmed": "citation",
    "children": "co-purchased or co-viewed",
    "history": "co-purchased or co-viewed",
    "computer": "co-purchased or co-viewed",
    "photo": "co-purchased or co-viewed",
    "sports": "co-purchased or co-viewed",
    "instagram": "following",
    "reddit": "replying",

    "cora": "citation",
    "products": "co-purchased or co-viewed",
    "wikics": "joint edited",
    "cora_link": "citation",
}

URL_LIST_EVAL = {
    "eval_server_0": {
        "name": "eval_server_0",
        "model_name": "eval",
        "pool_size": 200,
        "url": "http://YOUR_NODE:PORT/v1/chat/completions",
        "status": "active"
    },
    # 可按需添加更多评估服务器端点
}


def resolve_repo_path(*parts: str) -> Path:
    return ROOT_DIR.joinpath(*parts)


def ensure_eval_graph_pt(dataset: str) -> Path:
    candidates = [
        resolve_repo_path("data", "test_data", "raw_data", dataset, f"{dataset}_graph_bert_encoded.pt"),
        resolve_repo_path("data", "train_data", "raw_data", dataset, f"{dataset}_graph_bert_encoded.pt"),
    ]
    for path in candidates:
        if path.exists():
            return path

    if dataset not in {"products", "wikics", "fb15k237"}:
        raise FileNotFoundError(f"Missing graph pt for {dataset}: checked {candidates}")

    builder = resolve_repo_path("data_generation", "pipelines", "rebuild_graph_bert_encoded.py")
    cmd = [
        sys.executable,
        str(builder),
        "--datasets",
        dataset,
        "--skip-arxiv",
    ]
    subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Failed to generate graph pt for {dataset}")

# 为 GOFA 测试集构造 SSR 推理 prompt
# 加载原始 CSV，提取图结构信息并重新拼接为标准 SSR prompt 格式
def get_gofa_prompt(data_path, group_size):
    from data_generation.pipelines.prompt_templete import construct_prompt_chosen_reasoning

    df = pd.read_csv(data_path)
    gofa_test_data = {}

    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"load test data"):
        prompt = str(row['prompt'])
        dataset = str(row['dataset'])
        task_type = str(row['task'])
        truth_label = str(row['ground_truth'])

        # 从原始 prompt 中用正则提取目标节点、节点信息、边信息和标签列表
        target_node_pattern = r'node\d+'
        target_node_matches = re.findall(target_node_pattern, prompt.split("\n")[0], re.DOTALL)
        target_nodes = ', '.join([match.strip() for match in target_node_matches])

        node_pattern = r'node\d+:.*?\n'
        node_matches = re.findall(node_pattern, prompt, re.DOTALL)
        node_information = '\n'.join(["  " + match.strip() for match in node_matches])

        edge_pattern = r'<node\d+, node\d+>'
        edge_matches = re.findall(edge_pattern, prompt, re.DOTALL)
        edge_information = ', '.join([match.strip() for match in edge_matches])

        if task_type == 'nc':
            label_pattern = r'Select strictly from:(.*?)\.'
            matches = re.findall(label_pattern, prompt, re.DOTALL)
            if "gofa_test_data_53114.csv" in data_path:
                label_list = [label.strip().strip("'") for label in matches[0].split("', '")]
            elif "gofa_supply_data_35603.csv" in data_path:
                label_list = [label.strip() for label in matches[0].split(";")]
            else:
                raise ValueError("Unknown data_path for nc task")
            if dataset == "cora_node" or dataset == "cora_node_2ways":
                label_list = [label.replace("Theroy", "Theory") for label in label_list]
        elif task_type == 'lc':
            label_pattern = r'Choose from the following:(.*?)\.'
            matches = re.findall(label_pattern, prompt, re.DOTALL)
            label_list = [label.strip().strip("'") for label in matches[0].split(';')]
        elif task_type == 'lp':
            label_pattern = r'select strictly from: (.*?)\.Respond only with the'
            matches = re.findall(label_pattern, prompt, re.DOTALL)
            label_list = [label.strip().strip("'") for label in matches[0].split("' or '")]
        else:
            label_list = []
        label_list = "<" + ">, <".join(label_list) + ">"

        if row['task'] not in ["nc", "lc"]:
            continue

        # 执行完整性检查：确保 prompt 中包含必要的组件（节点、边、标签）
        if len(target_nodes) == 0:
            raise ValueError("target_nodes is missing in the prompt.")
        if len(node_matches) == 0:
            raise ValueError("node_matches is missing in the prompt.")
        elif len(node_matches) >= 2:
            if len(edge_matches) == 0:
                raise ValueError("edge_matches is missing in the prompt.")
        if len(label_list) == 0:
            raise ValueError("label_list is missing in the prompt.")
        prompt = construct_prompt_chosen_reasoning(task_type, target_nodes, node_information, edge_information, label_list, group_size)

        # 将处理后的 prompt 和 truth label 按 (数据集+任务类型) 组织为结构化格式
        task_key = f"{dataset}+{task_type}"
        if task_key not in gofa_test_data:
            gofa_test_data[task_key] = {
                "messages": [[{"role": "user", "content": prompt}]],
                "truth_labels": [truth_label]
            }
        else:
            gofa_test_data[task_key]["messages"].append([{"role": "user", "content": prompt}])
            gofa_test_data[task_key]["truth_labels"].append(truth_label)

    for task_key in gofa_test_data:
        gofa_test_data[task_key]["answer_set"] = set(gofa_test_data[task_key]["truth_labels"])

    return gofa_test_data

# 定义每个数据集的合法答案集合
# 用于判断模型输出是否在预期范围内（valid answer vs unknown answer）
def get_answer_set(dataset):
    answer_set_dic = {  
        "cora_node": set(['rule learning', 'genetic algorithms', 'case-based', 'theory', 'neural networks', 'probabilistic methods', 'reinforcement learning']),
        "wikics": set(['operating systems', 'computer security', 'programming language topics', 'computational linguistics', 'distributed computing architecture', 'computer architecture', 'internet protocols', 'databases', 'web technology', 'computer file systems']),
        "products": set(['baby', 'camera & photo', 'magazine subscriptions', 'cds & vinyl', 'kindle store', 'computers', 'tools & home improvement', 'software', 'video games', 'books', 'musical instruments', 'movies & tv', 'label 25', 'kitchen & dining', 'toys & games', 'baby products', 'gps & navigation', 'gift cards', 'digital music', 'grocery & gourmet food', 'home improvement', 'furniture & decoration', 'health & personal care', 'all electronics', 'electronics', 'all beauty', '#508510', 'beauty', 'mp3 players & accessories', 'home & kitchen', 'collectibles & fine art', 'purchase circles', 'car electronics', 'office products', 'patio, lawn & garden', 'pet supplies', 'clothing, shoes & jewelry', 'arts, crafts & sewing', 'industrial & scientific', 'buy a kindle', 'amazon fashion', 'luxury beauty', 'sports & outdoors', 'cell phones & accessories', 'office & school supplies', 'automotive', 'appliances'])
    }
    answer_set_dic["cora_node_2ways"] = answer_set_dic["cora_node"]
    answer_set_dic["products_10ways"] = answer_set_dic["products"]
    answer_set_dic["products_5ways"] = answer_set_dic["products"]
    answer_set_dic["wikics_5ways"] = answer_set_dic["wikics"]

    if dataset in ["fb15k237"]:
        data_file = ensure_eval_graph_pt(dataset)
        data = torch.load(data_file, weights_only=False)
        answer_set = set([label.lower().strip() for label in data.label_list])
    elif dataset in ["cora_node", "cora_node_2ways", "products_10ways", "products_5ways", "wikics_5ways"]:
        answer_set = answer_set_dic[dataset]
    elif dataset in ["wikics", "products"]:
        data_file = ensure_eval_graph_pt(dataset)
        data = torch.load(data_file, weights_only=False)
        answer_set = set([label.lower().strip() for label in data.label_list])
        answer_set = answer_set | answer_set_dic[dataset]
    else:
        raise ValueError("Unsupported dataset")
    
    return answer_set

# 从模型响应中提取答案
# 优先从 <Answer> 标签中提取，如果格式错误则标记为 "wrong format"
def get_answer(response):
    try:
        if "</think>" in response:
            response = response.split("</think>", 1)[-1]

            Chosen_subgraph_reasoning_pattern = r"<Chosen_subgraph_reasoning>(.*?)</Chosen_subgraph_reasoning>"
            Chosen_subgraph_reasoning_match = re.findall(Chosen_subgraph_reasoning_pattern, response, re.DOTALL)
            Chosen_subgraph_reasoning = Chosen_subgraph_reasoning_match[-1].strip()

            pattern = r'<Answer>(.*?)</Answer>'
            match = re.findall(pattern, Chosen_subgraph_reasoning, re.DOTALL)
            ans = match[-1].strip()
            ans_type = "normal"
            return ans_type, ans
        else:
            ans_type = "wrong format"
            ans = ""
            return ans_type, ans
    except Exception as e:
        ans_type = "wrong format"
        ans = ""
        return ans_type, ans
    
# 检查模型答案的正确性
# 分类：correct answer / wrong answer / unknown answer / wrong format
def check_answer(ans_type, ans, truth_label, task_type, answer_set):
    if ans_type == "wrong format":
        ans_dic = {
            "ans_type": ans_type,
            "truth_label": truth_label,
            "ans": ans
        }
    else:
        ans = ans.lower().strip()
        if ans in answer_set:
            if ans == truth_label:
                ans_type = "correct answer"
            else:
                ans_type = "wrong answer"
        else:
            if ans in truth_label:
                ans_type = "correct answer"
            else:
                ans_type = "unknown answer"
        ans_dic = {
            "ans_type": ans_type,
            "truth_label": truth_label,
            "ans": ans
        }
    return ans_dic

# 收集所有模型响应中的答案并评估正确性
def test_answer(json_list, task_type, answer_set):
    answers = []
    for item in json_list:
        response = item["response"]
        index = item["index"]
        truth_label = item["truth_label"].lower().strip()
        ans_type, ans = get_answer(response)
        ans_dic = check_answer(ans_type, ans, truth_label, task_type, answer_set)
        ans_dic["index"] = index
        answers.append(ans_dic)
    return answers

# 获取数据集的合法答案集合后评估所有回答
def evaluate_answer(dataset, task_type, json_list):
    answer_set = get_answer_set(dataset)
    evaluated_answers = test_answer(json_list, task_type, answer_set)

    return evaluated_answers

# 基于所有答案计算评估指标
# 包括总体准确率、各标签的准确率、格式错误率等
def cal_metrics(answers):
    total = len(answers)
    correct_answer = sum(1 for ans in answers if ans["ans_type"] == "correct answer")
    wrong_answer = sum(1 for ans in answers if ans["ans_type"] == "wrong answer")
    unknown_answer = sum(1 for ans in answers if ans["ans_type"] == "unknown answer")
    wrong_format = sum(1 for ans in answers if ans["ans_type"] == "wrong format")

    label_metrics = {}
    for ans in answers:
        ans_type = ans["ans_type"]
        truth_label = ans["truth_label"]
        if truth_label not in label_metrics:
            label_metrics[truth_label] = {
                "total": 0,
                "correct": 0,
                "wrong_answer": 0,
                "unknown_answer": 0,
                "wrong_format": 0
            }
        
        label_metrics[truth_label]["total"] += 1
        if ans_type == "correct answer":
            label_metrics[truth_label]["correct"] += 1
        elif ans_type == "wrong answer":
            label_metrics[truth_label]["wrong_answer"] += 1
        elif ans_type == "unknown answer":
            label_metrics[truth_label]["unknown_answer"] += 1
        elif ans_type == "wrong format":
            label_metrics[truth_label]["wrong_format"] += 1

    for label in label_metrics:
        label_total = label_metrics[label]["total"]
        label_correct = label_metrics[label]["correct"]
        label_accuracy = label_correct / label_total if label_total > 0 else 0.0
        label_metrics[label]["accuracy"] = label_accuracy

    accuracy = correct_answer / total if total > 0 else 0.0
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

# 评估主函数：加载数据 → 推理 → 评估 → 保存结果
def main(data_path, save_dir, seed, temperature, choose_server, group_size):

    # 加载测试数据并构造评估 prompt
    print(f"Loading test data from {data_path}...")
    test_data = get_gofa_prompt(data_path, group_size)
    
    sampling_params = {
        "temperature": temperature,
        "max_tokens": 16384,
        "top_k": 50,
        "seed": seed
    }

    # 遍历测试数据中的每个任务
    for task_key in test_data:
        print(f"Processing task: {task_key}...")
        print(f"Number of samples: {len(test_data[task_key]['messages'])}")

        dataset, task_type = task_key.split('+')
        save_path = f'{save_dir}/{dataset}/{dataset}_{task_type}_inference_result.jsonl'
        metrics_save_path = f'{save_dir}/{dataset}/{dataset}_{task_type}_metrics_result.json'

        # 如果推理结果文件不存在，执行推理并保存
        if not os.path.exists(save_path):
            messages = test_data[task_key]["messages"]
            truth_labels = test_data[task_key]["truth_labels"]
            if os.path.exists(f'{save_dir}/{dataset}') is False:
                os.makedirs(f'{save_dir}/{dataset}')
            
            eval_api(save_path, messages, truth_labels, sampling_params, choose_server)

        print(f"Loading existing inference results from {save_path}...")
        json_list = []
        with open(save_path, 'r') as f:
            for line in f:
                json_list.append(json.loads(line.strip()))

        # 获取数据集的合法答案集合后评估所有回答 and calculate the evaluation metrics
        evaluated_answers = evaluate_answer(dataset, task_type, json_list)
        metrics = cal_metrics(evaluated_answers)
        print(f"Acc for {task_key}: {metrics['accuracy']*100:.4f}%")
        with open(metrics_save_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        with open(f'{save_dir}/{dataset}/{dataset}_{task_type}_evaluated_answers.jsonl', 'w') as f:
            for item in evaluated_answers:
                f.write(json.dumps(item) + '\n')

# 调用推理 API 对给定 prompt 执行推理并将结果保存到文件
# 多次重试直到格式合规率达到要求（最多 8 次）
def eval_api(save_path, messages, truth_labels, sampling_params, choose_server):
    from utils.utils import reason

    samples = []
    for i in range(len(messages)):
        row = {
            "index": i,
            "prompt": messages[i][0]['content'],
            "truth_label": truth_labels[i]
        }
        samples.append(row)

    choose_name = choose_server
    url_list = [URL_LIST_EVAL[name] for name in choose_name]

    trying_times = 0
    while True:
        print(f"Reasoning attempt number: {trying_times + 1}")

        save_path_reasoning = save_path.replace("_inference_result.jsonl", "_reasoning_result.jsonl")
        generated_sample_indicies = set()
        if os.path.exists(save_path_reasoning):  
            print(f"Loading generated samples from {save_path_reasoning}")
            with open(save_path_reasoning, encoding='utf-8') as f:
                for raw_line in f:
                    sample = json.loads(raw_line)
                    generated_sample_indicies.add(int(sample["index"]))

        print(f"{len(generated_sample_indicies)} generated samples")

        targets = [i for i in range(len(samples)) if i not in generated_sample_indicies]

        reason(samples, targets, save_path_reasoning, url_list, sampling_params, print_info=True)

        lines_new = []
        with open(save_path_reasoning, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line)
                ans_type, ans = get_answer(line["response"])
                if ans_type == "normal":
                    lines_new.append(line)
        if len(samples) - len(lines_new) <= len(samples) // 1000 or trying_times >= 8:
            break
        with open(save_path_reasoning, 'w', encoding='utf-8') as f:
            for line in lines_new:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        trying_times += 1

    with open(save_path_reasoning, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f.readlines()]

    lines = sorted(lines, key=lambda x: x['index'])
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-d', "--data_path", type=str, help="test data path")
    parser.add_argument('-s', "--save_dir", type=str, help="saving path")
    parser.add_argument('-e', "--seed", type=int, default=42, help="random seed")
    parser.add_argument('-t', "--temperature", type=float, default=0.5, help="temperature setting for inference")
    parser.add_argument('-c', "--choose_server", nargs='+', help='List of evaluation servers')
    parser.add_argument('-g', "--group_size", type=int, default=5, help="group size for subgraph sampling")

    args = parser.parse_args()
    data_path = args.data_path
    save_dir = args.save_dir
    seed = args.seed
    temperature = args.temperature
    choose_server = args.choose_server
    main(data_path, save_dir, seed, temperature, choose_server, args.group_size)
