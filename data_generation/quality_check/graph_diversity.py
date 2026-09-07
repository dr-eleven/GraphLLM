"""
GraphSSR 采样子图的结构多样性评分模块。

本文件实现的功能：
- 构建成对比较 prompt，将采样的子图两两配对进行比较。
- 调用多样性评估模型（diversity model）为每对子图分配 0~1 的相似度/多样性分数。
- 汇总所有配对的多样性分数，供后续 SFT 数据过滤使用。

在训练流程中的角色：
- 仅用于 SFT 数据筛选阶段。
- 确保用于 SFT 训练的教师模型推理轨迹覆盖了具有充分差异的候选子图，
  鼓励模型学习从不同结构角度分析图任务。
"""

import re
import os
import sys
import time
import json
import requests
from multiprocessing import Manager
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import OrderedDict

root_dir = "../.."
sys.path.append(os.path.abspath(root_dir))

URL_LIST_DV = {
    "diversity_model_0": {
        "name": "diversity_model_0",
        "model_name": "diversity_model",
        "pool_size": 500,
        "url": "http://YOUR_NODE:PORT/v1/chat/completions",
        "status": "active"
    },
    # 可按需添加更多多样性模型服务器端点
}

# 格式化子图信息用于多样性评分计算
# 将中心节点、邻居节点和连接关系拼接为统一的文本格式
def format_subgraph_info(neighboring_dic, central_node_ans, neighboring_node_ans, connection_relationship_ans):

    out_central_node_ans = ', '.join(central_node_ans)

    out_neighboring_node_ans = [f"{node}:{neighboring_dic.get(node, '')}" for node in central_node_ans] + [f"{node}:{neighboring_dic.get(node, '')}" for node in neighboring_node_ans]
    out_neighboring_node_ans = '\n'.join(["  " + node_information for node_information in out_neighboring_node_ans])

    out_connection_relationship_ans = ', '.join(connection_relationship_ans)

    return out_central_node_ans, out_neighboring_node_ans, out_connection_relationship_ans


def call_deepseek(messages: list, url: str, model_name: str, sampling_params: dict):
    sampling_params_default = {  
        "messages": messages,
        "model": model_name,
        "stop": None,
        "stream": False,
    }
    sampling_params_default.update(sampling_params)
    payload = json.dumps(sampling_params_default)
    
    headers = {  
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.post(url, headers=headers, data=payload, timeout=None)
    if response.status_code != 200:
        raise RuntimeError(f"status code: {response.status_code}")
    response_data = json.loads(response.text)
    res = response_data["choices"][0]["message"]["content"]
    return res, response_data["usage"]["completion_tokens"]


def worker(args):
    index, samples, url, model_name, sampling_params = args
    sample = samples[index]
    
    try:
        messages = [{"role": "user", "content": sample["prompt"]}]
        response_data, completion_tokens = call_deepseek(messages, url, model_name, sampling_params)
        sample["response"] = response_data
        sample["completion_tokens"] = completion_tokens
        return True, index, (sample, sample["completion_tokens"]), url
    except Exception as e:
        time.sleep(30)
        return False, index, str(e), url
    

def submit_tasks(url_list, status_dic, pool, futures, targets, target_index, shared_samples, sampling_params, print_info=True):
    for url_i in range(len(url_list)):
        url = url_list[url_i]["url"]
        name = url_list[url_i]["name"]
        pool_size = url_list[url_i]["pool_size"]
        model_name = url_list[url_i]["model_name"]
        running_tasks = status_dic[url]["running_tasks"]
        available_slots = pool_size - running_tasks
        if available_slots > 0:
            next_target_index = min(target_index + available_slots, len(targets))
            submit_num = next_target_index - target_index
            futures.extend([pool.submit(worker, (i, shared_samples, url, model_name, sampling_params)) for i in targets[target_index:next_target_index]])
            target_index += submit_num
            status_dic[url]["running_tasks"] += submit_num
        else:
            submit_num = 0

        if print_info:
            print(f"Submitting tasks to {name}: {submit_num} tasks.")
            print(f"Status - Name: {name}, Running Tasks: {status_dic[url]['running_tasks']}, Finished Tasks: {status_dic[url]['finished_tasks']}, Total Completion Tokens: {int(status_dic[url]['total_completion_tokens'])}, Avg Throughput: {int(status_dic[url]['avg_throughput'])}, Max Throughput: {int(status_dic[url]['max_throughput'])}")

    return futures, target_index, status_dic

# 多样性评分的批量推理函数（从 utils/utils.py 修改而来，去掉了文件写入功能）
def reason(samples: list, targets: list, url_list: list, sampling_params: dict, print_info=True):  
    manager = Manager()
    shared_samples = manager.list(samples)

    total_completion_tokens = 0
    start_time = time.time()
    max_throughput = 0.0
    avg_throughput = 0.0
    sample_per_second = 0.0
    cnt = 0

    whole_pool_size = sum([url_list[i]["pool_size"] for i in range(len(url_list))])
    pool = ThreadPoolExecutor(whole_pool_size)
    status_dic = {}
    for i in range(len(url_list)):
        url = url_list[i]["url"]
        status_dic[url] = {
            "running_tasks": 0,
            "finished_tasks": 0,
            "total_completion_tokens": 0.0,
            "avg_throughput": 0.0,
            "max_throughput": 0.0
        }
    futures = []
    score_batch = []

    while True:
        failed = []
        target_index = 0
        finished_num = 0
        futures, target_index, status_dic = submit_tasks(url_list, status_dic, pool, futures, targets, target_index, shared_samples, sampling_params, print_info)
        with tqdm(total=len(targets), desc="Processing") as pbar:
            while True:
                for future in as_completed(futures):
                    result = future.result()
                    futures.remove(future)
                    finished_num += 1
                    pbar.update(1)
                    if not result[0]:
                        failed.append(result[1])
                        print(f"index {result[1]} failed, reason: {result[2]}")
                    else:
                        sample, completion_tokens = result[2]
                        url = result[3]
                        total_completion_tokens += int(completion_tokens)
                        status_dic[url]["total_completion_tokens"] += int(completion_tokens)
                        cnt += 1
                        cur_time = time.time() 
                        avg_throughput = total_completion_tokens / (cur_time - start_time)  
                        status_dic[url]["avg_throughput"] = status_dic[url]["total_completion_tokens"] / (cur_time - start_time)
                        if max_throughput < avg_throughput:  
                            max_throughput = avg_throughput  
                        status_dic[url]["max_throughput"] = max(status_dic[url]["max_throughput"], status_dic[url]["avg_throughput"])
                        sample_per_second = cnt / (cur_time - start_time)  
                        pbar.set_postfix(OrderedDict(avg_throughput=avg_throughput, max_throughput=max_throughput, sample_per_second=sample_per_second))
                        status_dic[url]["running_tasks"] -= 1
                        status_dic[url]["finished_tasks"] += 1

                        score_batch.append(sample)

                    if finished_num % 10 == 0:
                        futures, target_index, status_dic = submit_tasks(url_list, status_dic, pool, futures, targets, target_index, shared_samples, sampling_params, print_info)
                        break

                if len(futures) == 0 and target_index >= len(targets):
                    break

        if not failed:
            break
        targets = failed
        print(f"Last round failed {len(failed)} times")
    
    return score_batch

# 计算采样子图之间的多样性评分
# 对每条样本的所有子图两两配对，构造比较 prompt 并调用多样性模型评分
def get_diversity_score(prompt_batch, sampled_subgraph_list_batch, sample_index_batch):

    samples = []
    index = 0
    for si, sampled_subgraph_list in enumerate(sampled_subgraph_list_batch):

        # 从 prompt 中提取邻居节点信息（node_id → node_text 映射）
        node_pattern = r'node\d+:.*?\n'
        node_matches = re.findall(node_pattern, prompt_batch[si], re.DOTALL)
        neighboring_nodes = [match.strip().split(":") for match in node_matches]
        neighboring_dic = {node_info[0]: node_info[1] for node_info in neighboring_nodes}

        for i in range(len(sampled_subgraph_list)):
            for j in range(i+1, len(sampled_subgraph_list)):
                # 提取子图对中的各项信息
                _, central_node_ans_i, neighboring_node_ans_i, connection_relationship_ans_i, _ = sampled_subgraph_list[i]
                _, central_node_ans_j, neighboring_node_ans_j, connection_relationship_ans_j, _ = sampled_subgraph_list[j]

                # 格式化两个子图的结构信息
                central_node_ans_i, neighboring_node_ans_i, connection_relationship_ans_i = format_subgraph_info(neighboring_dic, central_node_ans_i, neighboring_node_ans_i, connection_relationship_ans_i)
                central_node_ans_j, neighboring_node_ans_j, connection_relationship_ans_j = format_subgraph_info(neighboring_dic, central_node_ans_j, neighboring_node_ans_j, connection_relationship_ans_j)

                # 构造多样性评分的 prompt：要求模型分析两个子图的差异并给出 0~1 的分数
                prompt = f"Task Description:\nYou are a graph analysis expert. Given two graph structures, which includes the description of the central node(s), its neighboring nodes, and their connection relationships, please follow the following steps to operate the analysis:\n\nGraph structure 1:\n- Central node ID(s): {central_node_ans_i}\n- Node texts (central node(s) and neighboring nodes): [\n{neighboring_node_ans_i}\n]\n- Connection relationships: [\n  {connection_relationship_ans_i}\n]\nIf the node texts and connection relationships are empty, it indicates that these pieces of information do not exist, don't try to fabricate.\n\nGraph structure 2:\n- Central node ID(s): {central_node_ans_j}\n- Node texts (central node(s) and neighboring nodes): [\n{neighboring_node_ans_j}\n]\n- Connection relationships: [\n  {connection_relationship_ans_j}\n]\nIf the node texts and connection relationships are empty, it indicates that these pieces of information do not exist, don't try to fabricate.\n\nStep description:\n1. Analyze the differences between the two graph structures in terms of the central node(s), neighboring nodes, and connection relationships.\n2. Based on the analysis, provide a diversity score ranging from 0 to 1 to quantify how different these two graph structures are. A score of 0 indicates that the two graph structures are identical, while a score of 1 indicates that they are completely different.\n3. Provide a brief explanation for the assigned diversity score.\n\nAnswer format:\nYour response should be in forms as follows, no other outputs are needed:\n<Graph_diversity_score>\n  <Score>[Diversity score here]</Score>\n  <Reason>[Brief explanation here]</Reason>\n</Graph_diversity_score>"

                samples.append({
                    "sample_index": sample_index_batch[si],
                    "pair_index": [i, j],
                    "index": index,
                    "prompt": prompt
                })
                index += 1

    # 调用推理函数获取所有子图对的多样性评分
    choose_name = list(URL_LIST_DV.keys())
    url_list = [URL_LIST_DV[name] for name in choose_name]

    targets = [i for i in range(len(samples))]

    sampling_params = {
        "temperature": 0.0,
        "max_tokens": 4096
    }
    score_batch = reason(samples, targets, url_list, sampling_params, print_info=False)
    score_batch = sorted(score_batch, key=lambda x: x['index'])

    return score_batch

# 从多样性模型的响应中提取评分和理由
def extract_deverity_score(response):
    response = response.split("</think>", 1)[-1]

    subgraph_diversity_pattern = r"<Graph_diversity_score>(.*?)</Graph_diversity_score>"
    
    subgraph_diversity_match = re.findall(subgraph_diversity_pattern, response, re.DOTALL)
    if len(subgraph_diversity_match) == 0:
        subgraph_diversity = ""
    else:
        subgraph_diversity = subgraph_diversity_match[-1].strip()

    score_pattern = r"<Score>(.*?)</Score>"
    score_match = re.findall(score_pattern, subgraph_diversity, re.DOTALL)
    try:
        score = float(score_match[-1].strip())
    except:
        score = 0.0

    reason_pattern = r"<Reason>(.*?)</Reason>"
    reason_match = re.findall(reason_pattern, subgraph_diversity, re.DOTALL)
    try:
        reason = reason_match[-1].strip()
    except:
        reason = ""

    return score, reason


# 检查 SFT 推理数据中采样子图的多样性
# 加载已有评分、分批计算新评分、追加保存
def check_sft_reasoning_diversity():
    from data_generation.quality_check.quality_evaluator import extract_sampled_subgraph_info

    # 加载 SFT 推理数据
    data_path = f"{root_dir}/outputs/synthetic_data/sft_data/sft_reasoning.jsonl"
    with open(data_path, 'r', encoding='utf-8') as f:
        sft_reasoning_data = [json.loads(line.strip()) for line in f]

    # 加载已计算的多样性评分（支持增量计算）
    output_file_name = f"{root_dir}/outputs/synthetic_data/sft_data/sft_reasoning_diversity_score.jsonl"
    if not os.path.exists(output_file_name):
        diversity_score_data = []
    else:
        with open(output_file_name, 'r', encoding='utf-8') as f:
            diversity_score_data = [json.loads(line.strip()) for line in f]
    already_checked_index = set([line["sample_index"] for line in diversity_score_data])
    
    # 准备待计算多样性评分的批次
    prompt_batch = [sample["prompt"] for sample in sft_reasoning_data if sample["sample_index"] not in already_checked_index]
    sampled_subgraph_list_batch = [extract_sampled_subgraph_info(sample["response"]) for sample in sft_reasoning_data if sample["sample_index"] not in already_checked_index]
    sample_index_batch = [sample["sample_index"] for sample in sft_reasoning_data if sample["sample_index"] not in already_checked_index]

    # 调用 get_diversity_score 批量计算
    score_batch = get_diversity_score(prompt_batch, sampled_subgraph_list_batch, sample_index_batch)

    # 整理评分结果并追加写入 JSONL 文件
    data_dic = {si:[] for si in sample_index_batch}
    for sample in score_batch:
        sample_index = sample["sample_index"]
        score, reason = extract_deverity_score(sample["response"])
        sample["diversity_score"] = score
        sample["diversity_reason"] = reason

        data_dic[sample_index].append(sample)

    for sample_index, sample_list in data_dic.items():
        diversity_score_data.append({"sample_index": sample_index, "sample_list": sample_list})
    diversity_score_data = sorted(diversity_score_data, key=lambda x: x["sample_index"])

    with open(output_file_name, "w") as f:
        for line in diversity_score_data:
            f.write(json.dumps(line) + "\n")
