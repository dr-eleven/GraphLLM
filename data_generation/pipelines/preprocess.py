"""
GraphSSR 子图预处理入口模块。

本文件实现的功能：
- 使用 PyG (PyTorch Geometric) 的 NeighborLoader / LinkNeighborLoader 为节点分类、
  链接预测和链接分类三种任务采样局部子图。
- 对采样的子图进行节点索引随机重排（shuffle），并将紧凑的 `.pt` 文件存储到磁盘，
  供后续 prompt 构造阶段使用。

在训练流程中的角色：
- 仅用于 RL 数据准备阶段。
- 生成子图池，供 `rl_data_generator.py` 在构建 PPO 训练/验证数据时从中采样。
"""

import os
import sys
import torch
import random
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit

root_dir = "../.."
sys.path.append(os.path.abspath(root_dir))

# 为不同的图任务类型生成子图数据
# 支持三种任务类型：
# - nc (节点分类): 用 NeighborLoader 围绕每个节点采集 2 跳邻居
# - lp (链接预测): 先对边做随机划分，再用 LinkNeighborLoader 采样子图
# - lc (链接分类): 类似 lp 但用于边的类型分类任务
# 采样后的子图过滤掉过小（≤3 节点或≤6 边）和过大（≥50 节点）的结果，
# 并对节点索引随机重排以保证无偏性
def generate_subgraph_data(dataset, task_type, graph_data_path, subgraph_data_path):

    random.seed(42)

    # 加载完整图数据（已经过 BERT 编码和格式化处理）
    graph_data = torch.load(graph_data_path, weights_only=False)

    if task_type == 'nc':
        # 节点分类子图采样：围绕每个节点采集 2 跳邻居（每跳 10 个邻居）
        dataloader = NeighborLoader(
            graph_data,
            num_neighbors=[10,10],
            input_nodes=torch.arange(graph_data.num_nodes),
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
    elif task_type == 'lp':  # 链接预测任务:
        # 对特殊数据集使用更多邻居（更密集的采样）
        special_datasets = ["cora", "products"]
        if dataset in special_datasets:
            neighbors_list = [10, 10]
        else:
            neighbors_list = [5, 5]

        # 随机采样部分边用于构造链接预测任务
        _, _, sample_link_graph = RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            add_negative_train_samples=False,
            neg_sampling_ratio=1.0,
            disjoint_train_ratio=0,
            is_undirected=True
        )(graph_data)

        # 构建链接预测子图采样器
        dataloader = LinkNeighborLoader(
            data=sample_link_graph,
            num_neighbors=neighbors_list,
            edge_label_index=sample_link_graph.edge_label_index,
            edge_label=sample_link_graph.edge_label,
            neg_sampling_ratio=0,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
    elif task_type == 'lc':  # 链接分类任务:
        neighbors_list = [5, 5]

        # 随机采样部分边用于构造链接分类任务
        sample_link_graph, _, _ = RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            add_negative_train_samples=False,
            disjoint_train_ratio=0,
            is_undirected=False,
            key='edge_map',
            neg_sampling_ratio=0,
        )(graph_data)

        # 构建链接分类子图采样器
        dataloader = LinkNeighborLoader(
            data=sample_link_graph,
            num_neighbors=neighbors_list,
            edge_label_index=sample_link_graph.edge_map_index,
            edge_label=sample_link_graph.edge_map,
            neg_sampling_ratio=0.0,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    subgraph_data = []
    for i, data in enumerate(tqdm(dataloader)):
        # 过滤掉过小（≤3节点或≤6边）或过大（≥50节点）的子图
        if data.x.size(0) <= 3 or data.edge_index.size(1) <= 6:
            continue
        elif data.x.size(0) >= 50:
            continue
        
        # 对子图中的节点索引进行随机重排，避免模型学习到位置偏差
        n = data.x.size(0)
        shuffled_indices = torch.randperm(n)
        inverse_indices = torch.argsort(shuffled_indices)

        # 将子图数据按重排后的索引重新组织
        data.x = data.x[shuffled_indices]
        edge_index_shape = data.edge_index.size()
        data.edge_index = inverse_indices[data.edge_index.view(-1)].view(edge_index_shape)
        data.n_id = data.n_id[shuffled_indices]

        if task_type == 'nc' or task_type == 'lp':
            data.y = data.y[shuffled_indices]
            data.train_mask = data.train_mask[shuffled_indices]
            data.val_mask = data.val_mask[shuffled_indices]
            data.test_mask = data.test_mask[shuffled_indices]

        if task_type == 'lp' or task_type == 'lc':
            data.target_node = inverse_indices[:2]
        elif task_type == 'nc':
            data.target_node = inverse_indices[:1]
            
        subgraph_data.append(data)

    # 打印统计信息并保存子图数据
    print("Total number of subgraphs:", len(dataloader))
    print("Actual number of subgraphs generated:", len(subgraph_data))
    torch.save(subgraph_data, subgraph_data_path)
    return subgraph_data

if __name__ == "__main__":
    from data_generation.pipelines.graph_prepare import dataset_stat

    for (dataset, task_type) in dataset_stat.keys():
        raw_data_dir = f"../data/train_data/raw_data/{dataset}"
        processed_data_dir = f"../data/train_data/processed_data/{dataset}"

        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)

        graph_data_path = f"{raw_data_dir}/{dataset}_graph_bert_encoded.pt"
        subgraph_data_path = f"{processed_data_dir}/{dataset}_{task_type}_subgraph_bert_encoded.pt"

        # 生成子图数据（预生成的图数据也可从发布链接下载）
        generate_subgraph_data(dataset, task_type, graph_data_path, subgraph_data_path)
