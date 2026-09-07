"""
GraphSSR RL 数据生成流水线。

本文件实现的功能：
- 从 `preprocess.py` 准备的子图池中采样候选子图。
- 为每个采样的图任务构建 SSR 格式的 prompt。
- 多次查询 SFT 模型来评估任务难度（easy / medium / hard）。
- 导出 RL 训练和验证所用的 Parquet 文件，供 PPO/GRPO 训练使用。

在训练流程中的角色：
- RL 数据准备阶段，位于 SFT 和策略优化之间。
- 生成 `reinforcement_learning/run_rl_stage*.sh` 脚本所消费的训练/验证文件。
"""

import re
import os
import sys
import json
import time
import copy
import random
import pickle
import importlib
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

URL_LIST_RL = {
    "server_0": {
        "name": "server_0",
        "model_name": "sft",
        "pool_size": 200,
        "url": "http://YOUR_NODE:PORT/v1/chat/completions",
        "status": "active"
    }
}

root_dir = "../.."
sys.path.append(os.path.abspath(root_dir))

# 根据多次推理结果判断问题难度
# 用 SFT 模型对同一问题多次推理，根据正确答案的比例划分：
# - easy（简单）：正确率 ≥ 60%
# - medium（中等）：正确率 > 0 且 < 60%
# - hard（困难）：正确率 = 0%
def check_question_difficulty(reason_samples):
    from data_generation.quality_check.quality_evaluator import extract_chosen_subgraph_reasoning_info, check_chosen_answer

    # 检查每次推理结果，收集正确答案
    answer_list = []
    for reason_sample in reason_samples:
        response = reason_sample["response"]
        truth_label = reason_sample["truth_label"]
        chosen_subgraph_reasoning_info = extract_chosen_subgraph_reasoning_info(response)
        answer = check_chosen_answer(truth_label, chosen_subgraph_reasoning_info)
        answer_list.append(answer)

    # 根据多次推理中正确答案的比例来确定难度
    ratio = sum(answer_list) / len(answer_list)
    if ratio >= 0.6:
        return "easy"
    elif ratio > 0 and ratio < 0.6:
        return "medium"
    else:
        return "hard"

# 构造带有难度标签的 RL 训练数据
def construct_rl_training_data():
    from data_generation.pipelines.graph_prepare import generate_data_for_rl, dataset_stat
    from data_generation.pipelines.prompt_templete import construct_prompt_chosen_reasoning
    from utils.utils import reason

    # 加载已有的采样索引和已检查样本（支持断点续传）
    have_sampled_index_dic_path = f"{root_dir}/outputs/synthetic_data/rl_data/have_sampled_index_dic.pkl"
    checked_sample_dic_path = f"{root_dir}/outputs/synthetic_data/rl_data/checked_sample_dic.pkl"

    if os.path.exists(have_sampled_index_dic_path):
        with open(have_sampled_index_dic_path, "rb") as f:
            have_sampled_index_dic = pickle.load(f)
    else:
        have_sampled_index_dic = {key: [] for key in dataset_stat.keys()}

    if os.path.exists(checked_sample_dic_path):
        with open(checked_sample_dic_path, "rb") as f:
            checked_sample_dic = pickle.load(f)
    else:
        checked_sample_dic = {key: {"easy": [], "medium": [], "hard": []} for key in dataset_stat.keys()}

    # 主循环：持续生成数据直到各数据集各难度的样本数都达到配额要求
    while True:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Generating data for RL...")
        # 根据当前采样进度生成新的 RL 数据
        graph_info_dic = generate_data_for_rl(have_sampled_index_dic, checked_sample_dic)
        
        need_to_reason_list = []
        reason_index = 0
        for key in dataset_stat.keys():
            dataset, task = key

            # 获取新的图信息用于推理，并更新已采样索引
            sample_index, graph_info_list = graph_info_dic[key]
            have_sampled_index_dic[key].extend(sample_index)

            for graph_info in tqdm(graph_info_list, desc=f"Construct RL prompts for {dataset}-{task}"):
                target_nodes = graph_info['target_nodes']
                node_information = graph_info['node_information']
                edge_information = graph_info['edge_information']
                label_list = graph_info['label_list']
                truth_label = graph_info['truth_label']

                # 用 SSR 模板为当前样本构造 prompt
                prompt = construct_prompt_chosen_reasoning(task, target_nodes, node_information, edge_information, label_list)

                # 准备推理输入数据
                line = {
                    "index": reason_index,
                    "sample_index": graph_info['sample_index'],
                    "dataset": dataset,
                    "task": task,
                    "prompt": prompt,
                    "truth_label": truth_label,
                }
                need_to_reason_list.append(line)
                reason_index += 1

        # 调用多个 vLLM 服务器执行批量推理
        choose_name = ["server_0"]
        url_list = [URL_LIST_RL[name] for name in choose_name]

        targets = list(range(len(need_to_reason_list)))

        sampling_params = {
            "temperature": 0.5,
            "max_tokens": 16384
        }

        # 多次推理以评估难度（默认尝试 5 次）
        try_times = 5
        reason_list = []
        for tt in range(try_times):
            print(f"Reasoning try time {tt+1}/{try_times}")
            results = reason(need_to_reason_list, targets, None, url_list, sampling_params, print_info=False)
            results = sorted(results, key=lambda x: x["index"])  # type: ignore
            reason_list.append(results)
        
        # 整理多次推理结果，按样本组织为 [[si_0_attempts], [si_1_attempts], ...]
        reason_sample_list = []
        for j in range(len(need_to_reason_list)):
            reason_samples = []
            for i in range(try_times):
                reason_samples.append(reason_list[i][j])
            reason_sample_list.append(reason_samples)

        # 评估每个样本的难度并归类
        checked_sample_dic_this_round = {key: {"easy": [], "medium": [], "hard": []} for key in dataset_stat.keys()}
        for reason_samples in reason_sample_list: # type: ignore
            difficulty = check_question_difficulty(reason_samples)

            key = (reason_samples[0]["dataset"], reason_samples[0]["task"])
            sample = copy.deepcopy(reason_samples[0])
            del sample["index"]
            del sample["response"]
            del sample["completion_tokens"]
            checked_sample_dic_this_round[key][difficulty].append(sample)

        # 更新已检查样本字典并记录各难度级别的完成状态
        status_dic = {}
        for key in checked_sample_dic.keys():
            dataset, task = key
            if len(graph_info_dic[key][1]) == 0:
                status_dic[key] = {"easy": True, "medium": True, "hard": True}
            else:
                status_dic[key] = {"easy": False, "medium": False, "hard": False}

            for difficulty in ["easy", "medium", "hard"]:
                needed_num = dataset_stat[key][difficulty] - len(checked_sample_dic[key][difficulty])
                if needed_num > 0:
                    if len(checked_sample_dic_this_round[key][difficulty]) <= needed_num:
                        samples_to_add = checked_sample_dic_this_round[key][difficulty]
                    else:
                        samples_to_add = random.sample(checked_sample_dic_this_round[key][difficulty], needed_num)
                    checked_sample_dic[key][difficulty].extend(samples_to_add)
                    print(f"Dataset {dataset}-{task} need {needed_num} {difficulty} samples, added {len(samples_to_add)} samples this round.")
                status_dic[key][difficulty] = len(checked_sample_dic[key][difficulty]) >= dataset_stat[key][difficulty]

        # 保存 RL 数据到 JSONL 文件，同时持久化采样进度
        output_file_name = f"{root_dir}/outputs/synthetic_data/rl_data/rl_filter_prompt.jsonl"
        dir_path = os.path.dirname(output_file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(output_file_name, "w", encoding="utf-8") as f:
            for key in checked_sample_dic.keys():
                for difficulty in ["easy", "medium", "hard"]:
                    for sample in checked_sample_dic[key][difficulty]:
                        sample["difficulty"] = difficulty
                        f.write(json.dumps(sample) + "\n")

        with open(have_sampled_index_dic_path, "wb") as f:
            pickle.dump(have_sampled_index_dic, f)
        with open(checked_sample_dic_path, "wb") as f:
            pickle.dump(checked_sample_dic, f)

        if all([all(status_dic[key].values()) for key in status_dic.keys()]):
            break

# 将过滤后的 JSONL 数据转换为 Parquet 格式用于训练
def get_parquet_for_training():
    data_source = "graph_rl_data"
            
    jsonl_path = f"{root_dir}/outputs/synthetic_data/rl_data/rl_filter_prompt.jsonl"
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]

    df_list = []
    for line in data:
        if line["task"] in ["nc", "lp", "lc"]:
            df_list.append({
                "dataset": line["dataset"],
                "task": line["task"],
                "prompt": np.array([{"role": "user", "content": line["prompt"]}]),
                "ground_truth": line["truth_label"],
                "data_source": data_source
            })
    random.seed(42)
    random.shuffle(df_list)
    print(df_list[0]["prompt"][0]["content"])

    output_path = f"{root_dir}/outputs/synthetic_data/rl_data/train_{len(df_list)}_{data_source}.parquet"
    df_out = pd.DataFrame(df_list)
    df_out.to_parquet(output_path, index=False)
    print(len(df_out))

# 将现有测试数据转换为 Parquet 格式用于验证
def get_parquet_for_testing():
    from data_generation.pipelines.graph_prepare import extract_info_from_rl_data
    from data_generation.pipelines.prompt_templete import construct_prompt_chosen_reasoning

    data_source = "graph_rl_data"
    input_name = "test_1050.parquet"
    output_name = "test_{}_{}.parquet"

    # 加载并过滤测试数据（仅保留 nc 和 lc 任务）
    df = pd.read_parquet(f"{root_dir}/data/train_data/rl_data/{input_name}")
    df_filtered = df[df['task'].isin(["nc", "lc"])].reset_index(drop=True)

    df_list = []
    df_json_list = []
    for index, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"load test data"):

        # 从每一行提取图结构信息
        graph_info, skip_status = extract_info_from_rl_data(row)
        task = graph_info['task']
        dataset = graph_info['dataset']
        target_nodes = graph_info['target_nodes']
        node_information = graph_info['node_information']
        edge_information = graph_info['edge_information']
        label_list = graph_info['label_list']
        ground_truth = graph_info['ground_truth']
        reward_model = graph_info['reward_model']
        extra_info = graph_info['extra_info']

        if skip_status:
            continue
        
        prompt = construct_prompt_chosen_reasoning(task, target_nodes, node_information, edge_information, label_list)

        df_list.append({
            "dataset": dataset,
            "task": task,
            "prompt": np.array([{"role": "user", "content": prompt}]),
            "ground_truth": ground_truth,
            "data_source": data_source,
            "reward_model": reward_model,
            "extra_info": extra_info
        })

        df_json_list.append({
            "dataset": dataset,
            "task": task,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "data_source": data_source,
            "reward_model": reward_model,
            "extra_info": extra_info
        })

    output_path = f"{root_dir}/outputs/synthetic_data/rl_data/{output_name.format(len(df_list), data_source)}"
    df_out = pd.DataFrame(df_list)
    df_out.to_parquet(output_path, index=False)
    print(len(df_out))

    with open(output_path.replace(".parquet", ".jsonl"), "w") as f:
        for row in df_json_list:
            f.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    # 步骤 1：生成带有难度标签的 RL 训练数据
    construct_rl_training_data()

    # 步骤 2：将过滤后的数据转换为训练用 Parquet 格式
    get_parquet_for_training()

    # 步骤 3：将测试数据转换为验证用 Parquet 格式
    get_parquet_for_testing()
