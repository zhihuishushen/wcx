"""
数据加载器模块
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any


class DualLayerGraphDataset(Dataset):
    """
    双层图数据集类
    """
    
    def __init__(self, data_dir: str, vulnerability_types: List[str] = None):
        """
        Args:
            data_dir: 数据目录路径
            vulnerability_types: 漏洞类型列表
        """
        self.data_dir = data_dir
        self.vulnerability_types = vulnerability_types or [
            'reentrancy',
            'unchecked external call',
            'ether frozen',
            'ether strict equality'
        ]
        
        # 加载数据文件列表
        self.data_files = []
        if os.path.isdir(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.json'):
                    self.data_files.append(os.path.join(data_dir, filename))
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取数据样本"""
        filepath = self.data_files[idx]
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理目标图
        target_graph = self._process_graph(data.get('target_graph', {}))
        
        # 处理上层图
        upper_graphs = [
            self._process_graph(g) 
            for g in data.get('upper_graphs', [])
        ]
        
        # 处理标签
        labels = self._process_labels(data.get('labels', {}))
        
        # 处理伪标签
        pseudo_labels = self._process_labels(data.get('pseudo_labels', {}))
        
        # 处理跨层边
        cross_edges = data.get('cross_edges', [])
        
        # 构建返回字典
        sample = {
            'target_graph': target_graph,
            'upper_graphs': upper_graphs,
            'cross_edges': cross_edges,
            'labels': labels,
            'pseudo_labels': pseudo_labels,
            'contract_name': data.get('target_contract', ''),
            'has_upper_graph': data.get('has_upper_graph', len(upper_graphs) > 0),
            'suspicious_score': data.get('suspicious_score', 0.0)
        }
        
        return sample
    
    def _process_graph(self, graph: Dict) -> Dict:
        """处理图数据"""
        if not graph:
            return {
                'node_features': torch.tensor([], dtype=torch.float),
                'edge_index': torch.tensor([[], []], dtype=torch.long),
                'edge_type': torch.tensor([], dtype=torch.long),
                'num_nodes': 0
            }
        
        # 节点特征
        node_features = graph.get('node_features', [])
        if isinstance(node_features, list):
            node_features = torch.tensor(node_features, dtype=torch.float)
        else:
            node_features = torch.tensor([], dtype=torch.float)
        
        # 边索引
        edge_index = graph.get('edge_index', [])
        if isinstance(edge_index, list) and len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)
        
        # 边类型
        edge_type = graph.get('edge_type', [])
        if isinstance(edge_type, list):
            edge_type = torch.tensor(edge_type, dtype=torch.long)
        else:
            edge_type = torch.tensor([], dtype=torch.long)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'num_nodes': node_features.size(0) if node_features.numel() > 0 else 0
        }
    
    def _process_labels(self, labels: Dict) -> torch.Tensor:
        """处理标签"""
        label_tensor = torch.zeros(len(self.vulnerability_types), dtype=torch.float)
        
        for i, vuln_type in enumerate(self.vulnerability_types):
            label_tensor[i] = labels.get(vuln_type, 0)
        
        return label_tensor


def default_collate_fn(batch: List[Dict]) -> Dict:
    """
    默认collate函数
    
    处理不同大小的图数据
    """
    target_graphs = []
    upper_graphs_list = []
    cross_edges_list = []
    labels_list = []
    pseudo_labels_list = []
    contract_names = []
    has_upper_list = []
    suspicious_scores = []
    
    max_target_nodes = 0
    max_upper_nodes = 0
    
    for sample in batch:
        # 记录基本信息
        contract_names.append(sample['contract_name'])
        has_upper_list.append(sample['has_upper_graph'])
        suspicious_scores.append(sample['suspicious_score'])
        
        # 处理目标图
        target_graph = sample['target_graph']
        target_graphs.append(target_graph)
        max_target_nodes = max(max_target_nodes, target_graph['num_nodes'])
        
        # 处理上层图
        upper_graphs = sample['upper_graphs']
        upper_graphs_list.append(upper_graphs)
        for ug in upper_graphs:
            max_upper_nodes = max(max_upper_nodes, ug['num_nodes'])
        
        # 处理标签
        labels_list.append(sample['labels'])
        pseudo_labels_list.append(sample['pseudo_labels'])
        
        # 处理跨层边
        cross_edges_list.append(sample['cross_edges'])
    
    # 填充目标图
    padded_target_graphs = []
    for tg in target_graphs:
        num_nodes = tg['num_nodes']
        node_features = tg['node_features']
        
        if num_nodes > 0 and node_features.numel() > 0:
            padded_features = torch.cat([
                node_features,
                torch.zeros(max_target_nodes - num_nodes, node_features.size(1))
            ], dim=0)
        else:
            padded_features = torch.zeros(max_target_nodes, 0)
        
        padded_target_graphs.append({
            'node_features': padded_features,
            'edge_index': tg['edge_index'],
            'edge_type': tg['edge_type'],
            'num_nodes': num_nodes
        })
    
    # 堆叠标签
    batch_labels = torch.stack(labels_list, dim=0)
    batch_pseudo_labels = torch.stack(pseudo_labels_list, dim=0)
    batch_suspicious_scores = torch.tensor(suspicious_scores, dtype=torch.float)
    batch_has_upper = torch.tensor(has_upper_list, dtype=torch.bool)
    
    return {
        'target_graphs': padded_target_graphs,
        'upper_graphs_list': upper_graphs_list,
        'cross_edges_list': cross_edges_list,
        'labels': batch_labels,
        'pseudo_labels': batch_pseudo_labels,
        'contract_names': contract_names,
        'has_upper_graph': batch_has_upper,
        'suspicious_score': batch_suspicious_scores
    }


class DataLoader:
    """
    数据加载器封装
    """
    
    def __init__(self, dataset: DualLayerGraphDataset, 
                 batch_size: int = 16,
                 shuffle: bool = True,
                 num_workers: int = 0):
        """
        Args:
            dataset: 数据集
            batch_size: 批次大小
            shuffle: 是否打乱
            num_workers: 工作进程数
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=default_collate_fn
        )
    
    def __len__(self) -> int:
        return len(self.dataloader)
    
    def __iter__(self):
        return iter(self.dataloader)

