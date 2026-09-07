"""
GraphSSR 共享推理工具模块。

本文件实现的功能：
- 为教师模型、SFT 模型和评估模型提供多端点 HTTP 推理辅助函数。
- 提供重试机制、任务调度和结果排序等在整个项目中复用的工具函数。

在训练流程中的角色：
- SFT 数据生成阶段：分发教师模型的推理任务，将大量 prompt 并发发送到多个 vLLM 服务端点。
- RL 数据生成与评估阶段：作为通用的批量推理后端被复用，支持并发调用、自动重试和吞吐量监控。
"""

import os
import re
import sys
import time
import json
import requests
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
from tenacity import retry, stop_after_attempt
from multiprocessing import Manager
from concurrent.futures import ThreadPoolExecutor, as_completed

URL_LIST_DS = {
    "teacher_model_0": {
        "name": "teacher_model_0",
        "model_name": "teacher_model",
        "pool_size": 100,
        "url": "http://YOUR_NODE:PORT/v1/chat/completions",
        "status": "active"
    },
    # 可按需添加更多教师模型服务器端点
}

root_dir = ".."
sys.path.append(os.path.abspath(root_dir))

# 调用 vLLM 兼容的 HTTP API 端点（带重试机制），最多重试 3 次
@retry(stop=stop_after_attempt(3), reraise=True)
def call_deepseek(messages: list, url: str, model_name: str, sampling_params: dict):
    # 构建请求参数，默认不使用 stop token，不使用流式输出
    sampling_params_default = {
        "messages": messages,
        "model": model_name,
        "stop": None,
        "stream": False
    }
    sampling_params_default.update(sampling_params)
    payload = json.dumps(sampling_params_default)
    
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    # 发送 HTTP POST 请求，不设超时（推理可能耗时较长）
    response = requests.post(url, headers=headers, data=payload, timeout=None)
    if response.status_code != 200:
        raise RuntimeError(f"status code: {response.status_code}")
    response_data = json.loads(response.text)
    # 从 OpenAI 兼容的响应格式中提取生成的文本和 token 数
    res = response_data["choices"][0]["message"]["content"]
    return res, response_data["usage"]["completion_tokens"]

    
# 线程池中的 worker 函数：对单个样本执行推理任务
# 接收 (index, samples, url, model_name, sampling_params) 元组，
# 调用模型 API 获取推理结果，返回成功/失败状态和结果数据
def worker(args):
    index, samples, url, model_name, sampling_params = args
    sample = samples[index]
    
    try:
        messages = [{"role": "user", "content": sample["prompt"]}]
        start_time = time.time()
        # 调用 API 获取推理结果
        response_data, completion_tokens = call_deepseek(messages, url, model_name, sampling_params)
        end_time = time.time()
        latency = end_time - start_time
        sample["response"] = response_data
        sample["completion_tokens"] = completion_tokens
        sample["latency"] = latency
        return True, index, (sample, sample["completion_tokens"]), url
    except Exception as e:
        # 失败时等待 30 秒后返回失败状态，由上层调度器进行重试
        time.sleep(30)
        return False, index, str(e), url
    
# 将待处理的任务提交到可用的模型端点
# 根据每个端点的 pool_size 和当前正在运行的任务数，动态分配新任务以最大化吞吐量
def submit_tasks(url_list, status_dic, pool, futures, targets, target_index, shared_samples, sampling_params, print_info=True):
    # 计算所有模型端点的可用槽位总数
    all_available_slots = 0
    for url_i in range(len(url_list)):
        available_slots = url_list[url_i]["pool_size"] - status_dic[url_list[url_i]["url"]]["running_tasks"]
        all_available_slots += available_slots

    # 根据各端点的可用槽位，将任务均匀分配到各个模型服务器
    for url_i in range(len(url_list)):
        url = url_list[url_i]["url"]
        name = url_list[url_i]["name"]
        pool_size = url_list[url_i]["pool_size"]
        model_name = url_list[url_i]["model_name"]
        running_tasks = status_dic[url]["running_tasks"]
        available_slots = pool_size - running_tasks

        # 如果总可用槽位超过剩余任务数，按端点数量均分任务以避免某端点过载
        if all_available_slots > len(targets):
            available_slots = min(available_slots, len(targets) // len(url_list) if len(targets) % len(url_list) == 0 else len(targets) // len(url_list) + 1)

        # 将任务提交到线程池，每个任务由 worker 函数处理
        if available_slots > 0:
            next_target_index = min(target_index + available_slots, len(targets))
            submit_num = next_target_index - target_index
            futures.extend([pool.submit(worker, (i, shared_samples, url, model_name, sampling_params)) for i in targets[target_index:next_target_index]])
            target_index += submit_num
            status_dic[url]["running_tasks"] += submit_num
        else:
            submit_num = 0

        # 打印任务提交的统计信息，包括各端点的运行中/已完成任务数和吞吐量
        if print_info:
            print(f"Submitting tasks to {name}: {submit_num} tasks.")
            print(f"Status - Name: {name}, Running Tasks: {status_dic[url]['running_tasks']}, Finished Tasks: {status_dic[url]['finished_tasks']}, Total Completion Tokens: {int(status_dic[url]['total_completion_tokens'])}, Avg Throughput: {int(status_dic[url]['avg_throughput'])}, Max Throughput: {int(status_dic[url]['max_throughput'])}")

    return futures, target_index, status_dic

# 通过 HTTP API 并发调用执行批量推理
# 这是整个项目的核心推理调度函数：
# - 使用多进程 Manager 共享样本数据
# - 用线程池并发管理所有模型端点
# - 动态提交任务、收集结果，失败任务自动重试
# - 实时监控吞吐量（tokens/s、samples/s）
# - 结果可选地写入文件或返回为列表
def reason(samples, targets, output_file_name, url_list, sampling_params, print_info=True):
    # 使用 multiprocessing Manager 在进程间共享样本数据
    manager = Manager()
    shared_samples = manager.list(samples)

    total_completion_tokens = 0
    start_time = time.time()
    max_throughput = 0.0
    avg_throughput = 0.0
    sample_per_second = 0.0
    cnt = 0

    if output_file_name is None:
        output_list = []

    # 初始化线程池（总大小 = 所有端点 pool_size 之和）和每个端点的状态字典
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

    # 主处理循环：提交任务 → 收集结果 → 重试失败任务
    while True:
        failed = []
        target_index = 0
        finished_num = 0

        # 提交初始批次任务
        futures, target_index, status_dic = submit_tasks(url_list, status_dic, pool, futures, targets, target_index, shared_samples, sampling_params, print_info)

        # 处理已完成的任务，使用进度条跟踪
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
                        sample, completion_tokens = result[2]  # 提取样本和生成的 token 数
                        url = result[3]
                        # 更新全局和各端点的 token 计数及吞吐量指标
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

                        if output_file_name is not None:
                            # 如果指定了输出文件，将结果以 JSONL 格式追加写入
                            with open(output_file_name, 'a', encoding='utf-8') as f:
                                raw_line = json.dumps(sample, ensure_ascii=False)
                                f.write(raw_line + '\n')
                                f.flush()
                        else:
                            output_list.append(sample)  # type: ignore

                    # 每完成 10 个任务时检查是否有新的槽位，并提交新任务
                    if finished_num % 10 == 0:
                        futures, target_index, status_dic = submit_tasks(url_list, status_dic, pool, futures, targets, target_index, shared_samples, sampling_params, print_info)

                        break

                # 检查是否所有任务都已完成：无 pending future 且已提交完所有 target
                if len(futures) == 0 and target_index >= len(targets):
                    break

        # 如果本轮没有失败的任务，退出循环；否则重试失败的任务
        if not failed:
            break
        targets = failed
        print(f"Last round failed {len(failed)} times")

    if output_file_name is None:
        return output_list  # type: ignore
    else:
        print("File has been written:", output_file_name)

# 按 index 对推理结果文件进行排序
# 由于并发推理的结果可能乱序写入文件，此函数按 index 重新排序以确保后续处理的一致性
def sort_reason_result(data_path):
    with open(data_path) as f:
        lines = [json.loads(line) for line in f.readlines()]

    lines = sorted(lines, key=lambda x: x['index'])
    output_path = data_path
    with open(output_path, 'w') as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
