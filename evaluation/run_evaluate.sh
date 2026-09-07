#!/bin/bash

# =============================================================================
# GraphSSR 模型评测启动脚本 (run_evaluate.sh)
# =============================================================================
#
# 功能概述：
#   对训练完成的模型在 GOFA 基准测试集上执行推理评测，自动计算准确率
#   和格式合规率等指标。本脚本分两次运行，分别评测 GOFA 的主测试集
#   （53114 条样本）和补充测试集（35603 条样本）。
#
# 在训练流程中的角色：
#   训练后评估阶段——SFT 或 RL 训练完成后，将最终模型部署为 vLLM 服务后
#   执行本脚本，得到模型在零样本图推理任务上的最终性能指标。
#
# 前置条件：
#   1. 必须先通过 scripts/deploy_api.sh 将训练好的模型部署为 vLLM 服务
#   2. 必须在 evaluate_trained_model.py 的 URL_LIST_EVAL 中配置好服务地址
#   3. 必须已经下载好 GOFA 测试 CSV 文件和对应的 .pt 图编码文件
#      （参考 GRAPHSSR_REPRODUCTION_GUIDE_ZH.md 第 5.1 节）
#
# 输出位置：
#   - 推理结果：outputs/test_result/gofa_test_data_53114/
#   - 推理结果：outputs/test_result/gofa_supply_data_35603/
#   - 每个子目录下会生成：
#       * {dataset}_{task}_reasoning_result.jsonl  — 原始推理输出
#       * {dataset}_{task}_inference_result.jsonl   — 排序后的推理结果
#       * {dataset}_{task}_evaluated_answers.jsonl  — 逐条答案评估
#       * {dataset}_{task}_metrics_result.json      — 汇总评估指标
#
# 命令行参数说明（传给 evaluate_trained_model.py）：
#   -d  data_path     ：测试 CSV 文件路径
#   -s  save_dir      ：结果保存目录
#   -e  seed          ：随机种子（默认 42，控制推理的 temperature 采样）
#   -t  temperature   ：推理温度（0.5，适中的随机性）
#   -c  choose_server ：使用的评测服务器名称列表（从 URL_LIST_EVAL 的 key 中读取）
#   -g  group_size    ：子图采样分组大小（默认 5）
# =============================================================================

# 保存脚本自身的执行目录，用于拼接相对路径
EXEC_DIR=$(pwd)

# =============================================================================
# 第一次运行：评测 GOFA 主测试集（53,114 条样本）
# 涵盖数据集：cora_link, cora_node, products, expla_graph, scene_graph,
#            fb15k237, wikics
# 涵盖任务类型：lp（链接预测）、nc（节点分类）、gc（图分类）、lc（链接分类）
# =============================================================================
python evaluate_trained_model.py \
    -d $EXEC_DIR/../data/test_data/raw_data/gofa_test_data/gofa_test_data_53114.csv \
    -s $EXEC_DIR/../outputs/test_result/gofa_test_data_53114 \
    -e 42 \
    -t 0.5 \
    -c $(python -c "from evaluate_trained_model import URL_LIST_EVAL; print(' '.join(URL_LIST_EVAL.keys()))")

# =============================================================================
# 第二次运行：评测 GOFA 补充测试集（35,603 条样本）
# 涵盖数据集：cora_node_2ways, products_10ways, products_5ways, wikics_5ways
# 涵盖任务类型：nc（节点分类，均为 few-shot 变体）
# =============================================================================
python evaluate_trained_model.py \
    -d $EXEC_DIR/../data/test_data/raw_data/gofa_test_data/gofa_supply_data_35603.csv \
    -s $EXEC_DIR/../outputs/test_result/gofa_supply_data_35603 \
    -e 42 \
    -t 0.5 \
    -c $(python -c "from evaluate_trained_model import URL_LIST_EVAL; print(' '.join(URL_LIST_EVAL.keys()))")
