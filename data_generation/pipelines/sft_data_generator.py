"""
GraphSSR SFT 数据生成流水线。

本文件实现的功能：
- 从 GraphR1 SFT 样本构建 SSR 格式的 prompt。
- 调用教师模型（teacher model）生成推理轨迹。
- 对推理结果执行真实性、一致性、多样性和答案正确性四重质量检查。
- 导出最终的 `graph_sft_data.json` 数据集，供 LlamaFactory 进行 SFT 训练。

在训练流程中的角色：
- SFT 数据合成阶段，位于监督微调之前。
- 本文件是从原始 GraphR1 标注数据到 GraphSSR 的 SSR 风格指令数据的桥梁。
"""

import re
import os
import sys
import importlib
import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

root_dir = "../.."
sys.path.append(os.path.abspath(root_dir))

# 使用 SSR 模板构造带推理要求的 prompt
def construct_prompt_reasoning(task, target_nodes, node_information, edge_information, label_list):
    from data_generation.pipelines.prompt_templete import construct_prompt_chosen_reasoning

    prompt = construct_prompt_chosen_reasoning(task, target_nodes, node_information, edge_information, label_list)

    return prompt

# 构造 SFT prompt 列表
# 从 GraphR1 SFT 数据集提取图信息，为每条样本构造 SSR prompt 并保存到 JSONL 文件
def construct_prompt_for_sft():
    from data_generation.pipelines.graph_prepare import get_graph_from_graphr1_sft
    graph_info_list = get_graph_from_graphr1_sft(root_dir)  # 从 GraphR1 SFT 数据集获取图信息列表

    output_path = f"{root_dir}/outputs/synthetic_data/sft_data/sft_prompt.jsonl"
    folder_path = os.path.dirname(output_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    output_data_list = []
    for index, graph_info in enumerate(tqdm(graph_info_list, desc="Construct SFT prompts")):
        dataset = graph_info['dataset']
        task = graph_info['task']
        target_nodes = graph_info['target_nodes']
        node_information = graph_info['node_information']
        edge_information = graph_info['edge_information']
        label_list = graph_info['label_list']
        truth_label = graph_info['truth_label']

        prompt = construct_prompt_reasoning(task, target_nodes, node_information, edge_information, label_list)

        line = {
            "sample_index": index,
            "dataset": dataset,
            "task": task,
            "prompt": prompt,
            "truth_label": truth_label,
        }
        output_data_list.append(line)

    with open(output_path, "w") as f:
        for line in output_data_list:
            f.write(json.dumps(line) + "\n")

# 调用教师模型为 SFT 数据生成推理结果
def sft_data_reasoning():
    from utils.utils import reason, URL_LIST_DS, sort_reason_result

    # 加载之前构造好的 SFT prompt 数据
    input_file_path = f"{root_dir}/outputs/synthetic_data/sft_data/sft_prompt.jsonl"
    samples = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            row = json.loads(line.strip())
            row["index"] = i
            samples.append(row)

    # 配置教师模型的 HTTP API 端点列表
    choose_name = ["teacher_model_0", "teacher_model_1", "teacher_model_2", "teacher_model_3"]
    url_list = [URL_LIST_DS[name] for name in choose_name]

    # 加载已生成的样本索引，避免重复推理
    output_file_name = f"{root_dir}/outputs/synthetic_data/sft_data/sft_reasoning.jsonl"
    generated_sample_indicies = set()
    if os.path.exists(output_file_name):  
        print(f"Loading generated samples from {output_file_name}")
        with open(output_file_name, encoding='utf-8') as f:
            for raw_line in f:
                sample = json.loads(raw_line)
                generated_sample_indicies.add(int(sample["index"]))

    print(f"{len(generated_sample_indicies)} generated samples")

    # 准备当前需要推理的目标样本（排除已生成的部分）
    targets = [i for i in range(len(samples)) if i not in generated_sample_indicies]

    # 设置采样参数并执行批量推理
    sampling_params = {
        "temperature": 0.5,
        "max_tokens": 16384
    }
    reason(samples, targets, output_file_name, url_list, sampling_params)

    # 对推理结果按索引排序
    sort_reason_result(output_file_name)

# 基于质量检查过滤 SFT 数据。候选组必须先通过严格的五子图格式、
# 真实性、中心节点-only 和确定性结构多样性检查，之后再执行论文中的
# LLM 两两多样性阈值、选择一致性与答案检查。
def filter_sft_data():
    from data_generation.quality_check.quality_evaluator import (
        check_center_only_subgraph,
        check_chosen_answer,
        check_chosen_reasoning_format,
        check_chosen_subgraph_consist,
        check_real_subgraph,
        check_sampled_subgraph_format,
        check_subgraph_count_and_ids,
        check_subgraph_diversity,
        extract_chosen_subgraph_reasoning_info,
        extract_prompt_info,
        extract_sampled_subgraph_info,
    )

    # 加载 SFT 推理数据和多样性评分
    sft_reasoning_data_path = f"{root_dir}/outputs/synthetic_data/sft_data/sft_reasoning.jsonl"
    with open(sft_reasoning_data_path, 'r', encoding='utf-8') as f:
        sft_reasoning_data = [json.loads(line.strip()) for line in f]

    diversity_score_path = f"{root_dir}/outputs/synthetic_data/sft_data/sft_reasoning_diversity_score.jsonl"
    with open(diversity_score_path, 'r', encoding='utf-8') as f:
        diversity_score_data = [json.loads(line.strip()) for line in f]

    # 构建样本索引到多样性评分列表的映射字典
    diversity_score_dic = {}
    for sample in diversity_score_data:
        sample_index = sample["sample_index"]
        if sample_index not in diversity_score_dic:
            diversity_score_dic[sample_index] = []
        for s in sample["sample_list"]:
            diversity_score_dic[sample_index].append(s["diversity_score"])

    checked_output_list = []
    diversity_score_list = []
    format_correct_num = 0
    diversity_pass_num = 0
    answer_correct_num = 0
    for sample in tqdm(sft_reasoning_data, desc="Construct SFT data with reasoning"):
        # 从样本中提取 prompt 信息、采样子图列表和选中子图推理信息
        prompt_info = extract_prompt_info(sample["prompt"])
        sampled_subgraph_list = extract_sampled_subgraph_info(sample["response"])
        chosen_subgraph_reasoning_info = extract_chosen_subgraph_reasoning_info(sample["response"])

        # 严格执行完整 SSR 流程检查。
        count_status, _ = check_subgraph_count_and_ids(sampled_subgraph_list)
        candidate_format_status, _ = check_sampled_subgraph_format(
            sample["response"],
            sampled_subgraph_list,
        )
        chosen_format_status, _ = check_chosen_reasoning_format(sample["response"])
        real_subgraph_status, _ = check_real_subgraph(prompt_info, sampled_subgraph_list)
        center_only_status, _ = check_center_only_subgraph(prompt_info, sampled_subgraph_list)
        rule_diversity_status, _, _ = check_subgraph_diversity(
            prompt_info,
            sampled_subgraph_list,
        )
        chosen_subgraph_consist_status, _ = check_chosen_subgraph_consist(
            sampled_subgraph_list,
            chosen_subgraph_reasoning_info,
        )
        answer_status = check_chosen_answer(
            sample["truth_label"],
            chosen_subgraph_reasoning_info,
        )
        valid_group_status = (
            count_status
            and candidate_format_status
            and real_subgraph_status
            and center_only_status
            and rule_diversity_status
        )

        # LLM 结构多样性评估：仅对完整、真实且一致的 SSR 样本计算。
        if valid_group_status and chosen_format_status and chosen_subgraph_consist_status:
            format_correct_num += 1
            diversity_score = diversity_score_dic[sample["sample_index"]]
            diversity_score = sum(diversity_score) / len(diversity_score) if len(diversity_score) > 0 else 0.0
            diversity_score_list.append(diversity_score)
            if diversity_score > 0.3:  # 多样性阈值设为 0.3
                diversity_pass_num += 1
            # 如果设置为 True，则忽略多样性过滤
                if answer_status:
                    checked_output_list.append(sample)
                    answer_correct_num += 1

    print(f"Total samples: {len(sft_reasoning_data)}")
    print(f"Format correct samples: {format_correct_num}")
    print(f"Format correct ratio: {format_correct_num / len(sft_reasoning_data):.4f}")
    print(f"Diversity pass samples: {diversity_pass_num}")
    print(f"Diversity pass ratio: {diversity_pass_num / len(sft_reasoning_data):.4f}")
    print(f"Answer correct samples: {answer_correct_num}")
    print(f"Answer correct ratio: {answer_correct_num / len(sft_reasoning_data):.4f}")

    # 保存过滤后的 SFT 推理数据和对应的多样性评分
    checked_output_index = set()
    with open(sft_reasoning_data_path, 'w', encoding='utf-8') as f:
        for sample in checked_output_list:
            f.write(json.dumps(sample) + "\n")
            checked_output_index.add(sample["sample_index"])

    with open(diversity_score_path, 'w', encoding='utf-8') as f:
        for line in diversity_score_data:
            if line["sample_index"] in checked_output_index:
                f.write(json.dumps(line) + "\n")

    return answer_correct_num / len(sft_reasoning_data)

# 循环执行推理 → 多样性检查 → 质量过滤，直到答案正确率达到 90%
def reason_check_filter_sft_data():
    from data_generation.quality_check.graph_diversity import check_sft_reasoning_diversity

    while True:
        sft_data_reasoning()
        time.sleep(2)
        check_sft_reasoning_diversity()
        time.sleep(2)
        correct_ratio = filter_sft_data()

        if correct_ratio >= 0.9:
            print(correct_ratio)
            print("SFT data generation completed.")
            break

# 将最终的 SFT 推理数据构造为 LlamaFactory 格式
def construct_sft_data():
    sft_reasoning_data_path = f"{root_dir}/outputs/synthetic_data/sft_data/sft_reasoning.jsonl"
    with open(sft_reasoning_data_path, 'r', encoding='utf-8') as f:
        sft_reasoning_data = [json.loads(line.strip()) for line in f]

    output_path = f"{root_dir}/outputs/synthetic_data/sft_data/graph_sft_data.json"
    json.dump(sft_reasoning_data, open(output_path, 'w', encoding='utf-8'), indent=2)


if __name__ == "__main__":
    # 步骤 1：从 GraphR1 SFT 数据构造 prompt
    construct_prompt_for_sft()

    # 步骤 2：教师模型推理 + 多样性检查 + 质量过滤（循环直到正确率 ≥ 90%）
    reason_check_filter_sft_data()

    # 步骤 3：构造最终的 SFT 训练数据
    construct_sft_data()
