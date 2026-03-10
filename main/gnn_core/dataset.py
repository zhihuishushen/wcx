"""
数据加载器模块
从Dataset目录加载智能合约数据
"""

import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset, DataLoader


def collate_fn(batch: List[Dict]) -> Dict:
    """
    自定义collate函数，处理变长图
    
    Args:
        batch: 样本列表
        
    Returns:
        批次字典
    """
    # 找出最大节点数
    max_nodes = max(item['node_features'].shape[0] for item in batch)
    
    # 初始化批次数组
    batch_size = len(batch)
    
    node_features = torch.zeros(batch_size, max_nodes, 215)
    edge_indices = []
    edge_types = []
    num_nodes_list = []
    labels = []
    inheritance_features = []
    
    for i, item in enumerate(batch):
        num_nodes = item['num_nodes']
        num_nodes_list.append(num_nodes)
        
        # 节点特征
        node_features[i, :num_nodes] = item['node_features'][:num_nodes]
        
        # 边（需要相对索引转换）
        if item['edge_index'].shape[1] > 0:
            edge_index = item['edge_index'].clone()
            edge_index[0] += i * max_nodes
            edge_index[1] += i * max_nodes
            edge_indices.append(edge_index)
            edge_types.append(item['edge_type'])
        
        # 标签
        labels.append(item['label'])
        
        # 继承特征
        inheritance_features.append(item['inheritance_features'])
    
    # 合并边
    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
        edge_type = torch.cat(edge_types, dim=0)
    else:
        edge_index = torch.zeros((2, 1), dtype=torch.long)
        edge_type = torch.zeros((1,), dtype=torch.long)
    
    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_type': edge_type,
        'num_nodes': torch.tensor(num_nodes_list, dtype=torch.long),
        'labels': torch.stack(labels),
        'inheritance_features': torch.stack(inheritance_features),
    }


# 数据集根目录 (指向sc/Dataset)
DATASET_ROOT = Path(__file__).parent.parent.parent.parent / "Dataset"

# 漏洞类型配置
VULNERABILITY_TYPES = {
    'reentrancy': 'reentrancy.csv',
    'timestamp dependency': 'timestamp dependency.csv',
    'unchecked external call': 'unchecked external call.csv',
    'ether frozen': 'ether frozen.csv',
    'ether strict equality': 'ether strict equality.csv',
    'integer overflow': 'integer overflow.csv',
    'dangerous delegatecall': 'dangerous delegatecall.csv',
    'block number dependency': 'block number dependency.csv',
}


@dataclass
class ContractInfo:
    """合约信息"""
    file_id: int
    contract_name: str
    label: int  # 0 or 1
    sol_path: str = ""
    source_code: str = ""
    
    # 提取的特征
    num_functions: int = 0
    num_lines: int = 0
    has_ether_transfer: bool = False
    has_delegatecall: bool = False
    has_reentrancy_pattern: bool = False
    has_call_value: bool = False
    has_payable: bool = False
    inheritance_depth: int = 0
    parent_contracts: List[str] = field(default_factory=list)
    
    # 图结构
    graph_nodes: List[str] = field(default_factory=list)
    graph_edges: List[Tuple[int, int, int]] = field(default_factory=list)  # (src, dst, edge_type)
    node_features: np.ndarray = None


class DatasetLoader:
    """
    数据加载器
    负责从Dataset目录加载智能合约数据
    """
    
    # 节点特征维度 (与GNNSCVulDetector一致)
    NODE_FEATURE_DIM = 215
    
    # 边类型映射
    EDGE_TYPES = {
        'call': 0,
        'create': 1,
        'send': 2,
        'delegatecall': 3,
        'inherit': 4,
        'import': 5,
        'variable': 6,
        'other': 7,
    }
    
    def __init__(self, dataset_root: str = None):
        """
        初始化数据加载器
        
        Args:
            dataset_root: 数据集根目录路径
        """
        self.dataset_root = Path(dataset_root) if dataset_root else DATASET_ROOT
        
    def load_vulnerability_csv(self, vuln_type: str) -> pd.DataFrame:
        """
        加载漏洞CSV文件
        
        Args:
            vuln_type: 漏洞类型
            
        Returns:
            DataFrame with columns: file, contract, ground truth
        """
        csv_file = self.dataset_root / VULNERABILITY_TYPES.get(vuln_type, f"{vuln_type}.csv")
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        return df
    
    def load_sol_file(self, vuln_type: str, file_id: int, contract_name: str) -> str:
        """
        加载.sol文件的源代码
        
        Args:
            vuln_type: 漏洞类型
            file_id: 文件ID
            contract_name: 合约名称
            
        Returns:
            合约源代码
        """
        # 处理目录名中的空格
        dir_name = vuln_type.replace(' ', ' ')
        sol_dir = self.dataset_root / dir_name
        
        # 尝试多种文件名格式
        possible_names = [
            f"{file_id}.sol",
            f"{file_id}_{contract_name}.sol",
            f"file_{file_id}.sol",
        ]
        
        for name in possible_names:
            sol_path = sol_dir / name
            if sol_path.exists():
                with open(sol_path, 'r', encoding='utf-8') as f:
                    return f.read()
        
        return ""
    
    def parse_solidity_features(self, source_code: str) -> Dict:
        """
        解析Solidity源代码，提取特征
        
        Args:
            source_code: Solidity源代码
            
        Returns:
            特征字典
        """
        features = {
            'num_functions': 0,
            'num_lines': 0,
            'has_ether_transfer': False,
            'has_delegatecall': False,
            'has_reentrancy_pattern': False,
            'has_call_value': False,
            'has_payable': False,
            'inheritance_depth': 0,
            'parent_contracts': [],
            'num_state_variables': 0,
            'has_modifier': False,
            'has_assert': False,
            'has_require': False,
        }
        
        if not source_code:
            return features
        
        lines = source_code.split('\n')
        features['num_lines'] = len(lines)
        
        # 提取函数定义
        functions = re.findall(r'function\s+(\w+)\s*\(', source_code)
        features['num_functions'] = len(functions)
        
        # 检查是否有ether转账
        features['has_ether_transfer'] = bool(
            re.search(r'(transfer|send|call\.value)', source_code)
        )
        
        # 检查是否有delegatecall
        features['has_delegatecall'] = bool(
            re.search(r'delegatecall', source_code, re.IGNORECASE)
        )
        
        # 检查重入漏洞模式
        features['has_reentrancy_pattern'] = bool(
            re.search(r'(transfer|send|call\.value).*(_|after|check)', source_code, re.DOTALL)
        )
        
        # 检查call.value
        features['has_call_value'] = bool(
            re.search(r'\.value\(', source_code)
        )
        
        # 检查payable函数
        features['has_payable'] = bool(
            re.search(r'payable', source_code)
        )
        
        # 提取继承关系
        inheritance = re.findall(r'contract\s+\w+\s+is\s+([\w,\s]+)', source_code)
        if inheritance:
            features['inheritance_depth'] = len([x.strip() for x in inheritance[0].split(',')])
            features['parent_contracts'] = [x.strip() for x in inheritance[0].split(',')]
        
        # 检查状态变量
        features['num_state_variables'] = len(
            re.findall(r'(uint|int|address|bool|string|bytes)\s+\w+', source_code)
        )
        
        # 检查modifier
        features['has_modifier'] = bool(
            re.search(r'modifier\s+\w+', source_code)
        )
        
        # 检查assert/require
        features['has_assert'] = bool(
            re.search(r'\bassert\s*\(', source_code)
        )
        features['has_require'] = bool(
            re.search(r'\brequire\s*\(', source_code)
        )
        
        return features
    
    def build_contract_graph(self, source_code: str) -> Tuple[List[str], List[Tuple[int, int, int]], np.ndarray]:
        """
        构建合约的代码语义图
        
        Args:
            source_code: Solidity源代码
            
        Returns:
            (nodes, edges, node_features)
        """
        if not source_code:
            return [], [], np.zeros((0, self.NODE_FEATURE_DIM))
        
        # 提取节点（函数和控制流）
        nodes = []
        edges = []
        
        # 函数节点
        functions = re.findall(r'function\s+(\w+)\s*\([^)]*\)', source_code)
        for func in functions:
            nodes.append(f"F_{func}")
        
        # 事件节点
        events = re.findall(r'event\s+(\w+)\s*\(', source_code)
        for event in events:
            nodes.append(f"E_{event}")
        
        # 变量节点
        variables = re.findall(r'(uint|int|address|bool|string|bytes)(\d*)\s+(\w+)', source_code)
        for var in variables:
            nodes.append(f"V_{var[2]}")
        
        # 如果没有节点，创建一个默认节点
        if not nodes:
            nodes = ["default_node"]
        
        # 提取边（调用关系）
        # 函数调用
        for func in functions:
            # 检查对其他函数的调用
            for called_func in functions:
                if called_func != func:
                    # 检查func是否调用了called_func
                    func_pattern = rf'function\s+{func}\s*\([^)]*\)[^{{]*{{[^}}]*\b{called_func}\b'
                    if re.search(func_pattern, source_code):
                        edges.append((nodes.index(f"F_{func}"), nodes.index(f"F_{called_func}"), self.EDGE_TYPES['call']))
        
        # 生成节点特征 (one-hot编码)
        node_features = self._generate_node_features(nodes, source_code)
        
        return nodes, edges, node_features
    
    def _generate_node_features(self, nodes: List[str], source_code: str) -> np.ndarray:
        """
        生成节点特征向量
        
        Args:
            nodes: 节点列表
            source_code: 源代码
            
        Returns:
            节点特征矩阵 (num_nodes, feature_dim)
        """
        num_nodes = len(nodes)
        features = np.zeros((num_nodes, self.NODE_FEATURE_DIM))
        
        # 基础特征: 节点类型
        for i, node in enumerate(nodes):
            if node.startswith('F_'):  # 函数
                features[i, 0] = 1
            elif node.startswith('E_'):  # 事件
                features[i, 1] = 1
            elif node.startswith('V_'):  # 变量
                features[i, 2] = 1
            else:  # 默认节点
                features[i, 3] = 1
            
            # 检查是否有external call
            node_code = node.split('_', 1)[-1] if '_' in node else node
            if node_code in source_code and '.call' in source_code:
                features[i, 4] = 1
            
            # 检查是否有状态修改
            if node_code in source_code and '=' in source_code.split(node_code, 1)[1].split('\n')[0]:
                features[i, 5] = 1
        
        # 如果特征维数超过NODE_FEATURE_DIM，截断
        if features.shape[1] > self.NODE_FEATURE_DIM:
            features = features[:, :self.NODE_FEATURE_DIM]
        
        return features
    
    def load_dataset(self, vuln_type: str, max_samples: int = None) -> List[ContractInfo]:
        """
        加载整个数据集
        
        Args:
            vuln_type: 漏洞类型
            max_samples: 最大样本数（用于测试）
            
        Returns:
            合约信息列表
        """
        df = self.load_vulnerability_csv(vuln_type)
        
        if max_samples:
            df = df.head(max_samples)
        
        contracts = []
        total = len(df)
        
        for idx, row in enumerate(df.iterrows()):
            if idx % 100 == 0:
                print(f"  Loading contracts: {idx}/{total} ({100*idx/total:.1f}%)", flush=True)
            
            file_id = row[1]['file']
            contract_name = row[1]['contract']
            label = row[1]['ground truth']
            
            # 加载源代码
            source_code = self.load_sol_file(vuln_type, file_id, contract_name)
            
            # 解析特征
            features = self.parse_solidity_features(source_code)
            
            # 构建图
            graph_nodes, graph_edges, node_features = self.build_contract_graph(source_code)
            
            contract = ContractInfo(
                file_id=file_id,
                contract_name=contract_name,
                label=label,
                sol_path=str(self.dataset_root / vuln_type / f"{file_id}.sol"),
                source_code=source_code,
                num_functions=features['num_functions'],
                num_lines=features['num_lines'],
                has_ether_transfer=features['has_ether_transfer'],
                has_delegatecall=features['has_delegatecall'],
                has_reentrancy_pattern=features['has_reentrancy_pattern'],
                inheritance_depth=features['inheritance_depth'],
                parent_contracts=features['parent_contracts'],
                graph_nodes=graph_nodes,
                graph_edges=graph_edges,
                node_features=node_features,
            )
            
            contracts.append(contract)
        
        print(f"  Loading contracts: {total}/{total} (100.0%)", flush=True)
        return contracts
    
    def get_dataset_statistics(self, contracts: List[ContractInfo]) -> Dict:
        """
        获取数据集统计信息
        
        Args:
            contracts: 合约列表
            
        Returns:
            统计信息字典
        """
        total = len(contracts)
        positive = sum(1 for c in contracts if c.label == 1)
        negative = total - positive
        
        # 继承关系统计
        has_inheritance = sum(1 for c in contracts if c.inheritance_depth > 0)
        
        # 图统计
        avg_nodes = np.mean([len(c.graph_nodes) for c in contracts]) if contracts else 0
        avg_edges = np.mean([len(c.graph_edges) for c in contracts]) if contracts else 0
        
        return {
            'total': total,
            'positive': positive,
            'negative': negative,
            'positive_ratio': positive / total if total > 0 else 0,
            'has_inheritance': has_inheritance,
            'avg_nodes': avg_nodes,
            'avg_edges': avg_edges,
        }


class ContractGraphDataset(Dataset):
    """
    PyTorch数据集类 - 支持变长图
    """
    
    # 节点特征维度
    NODE_FEATURE_DIM = 215
    
    def __init__(self, contracts: List[ContractInfo], vuln_type: str, max_nodes: int = 500):
        """
        初始化数据集
        
        Args:
            contracts: 合约信息列表
            vuln_type: 漏洞类型
            max_nodes: 最大节点数（用于padding）
        """
        self.contracts = contracts
        self.vuln_type = vuln_type
        self.max_nodes = max_nodes
        
        # 类别权重（用于处理不平衡）
        num_positive = sum(1 for c in contracts if c.label == 1)
        num_negative = len(contracts) - num_positive
        self.class_weights = torch.tensor([
            len(contracts) / (2 * num_negative) if num_negative > 0 else 1.0,
            len(contracts) / (2 * num_positive) if num_positive > 0 else 1.0
        ], dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.contracts)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个样本
        
        Returns:
            字典包含:
            - node_features: 节点特征 (max_nodes, feature_dim) - padding后的
            - edge_index: 边索引 (2, num_edges)
            - edge_type: 边类型 (num_edges,)
            - num_nodes: 实际节点数
            - label: 标签 (1,)
            - contract_info: 合约信息
        """
        contract = self.contracts[idx]
        
        # 获取实际节点数
        num_nodes = min(len(contract.graph_nodes), self.max_nodes)
        
        # 裁剪或padding节点特征
        if contract.node_features is not None and len(contract.node_features) > 0:
            actual_nodes = min(len(contract.node_features), self.max_nodes)
            node_features = torch.zeros((self.max_nodes, self.NODE_FEATURE_DIM), dtype=torch.float32)
            node_features[:actual_nodes] = torch.tensor(
                contract.node_features[:actual_nodes],
                dtype=torch.float32
            )
        else:
            node_features = torch.zeros((self.max_nodes, self.NODE_FEATURE_DIM), dtype=torch.float32)
        
        # 构建边索引（只保留在有效节点范围内的边）
        if contract.graph_edges:
            valid_edges = [(e[0], e[1], e[2]) for e in contract.graph_edges 
                          if e[0] < num_nodes and e[1] < num_nodes]
            if valid_edges:
                edge_index = torch.tensor(
                    [[e[0] for e in valid_edges],
                     [e[1] for e in valid_edges]],
                    dtype=torch.long
                )
                edge_type = torch.tensor(
                    [e[2] for e in valid_edges],
                    dtype=torch.long
                )
            else:
                edge_index = torch.zeros((2, 1), dtype=torch.long)
                edge_type = torch.zeros((1,), dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 1), dtype=torch.long)
            edge_type = torch.zeros((1,), dtype=torch.long)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'num_nodes': num_nodes,
            'label': torch.tensor(contract.label, dtype=torch.float32),
            'contract_name': contract.contract_name,
            'file_id': contract.file_id,
            # 继承相关性特征（用于预测器）
            'inheritance_features': torch.tensor([
                contract.inheritance_depth,
                contract.num_functions,
                contract.num_lines / 1000.0,  # 归一化
                float(contract.has_delegatecall),
                float(contract.has_ether_transfer),
                float(contract.has_reentrancy_pattern),
                float(contract.has_call_value),
                float(contract.has_payable),
            ], dtype=torch.float32),
        }


def create_data_loaders(
    vuln_type: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 16,
    max_samples: int = None,
    max_nodes: int = 500,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练/验证/测试数据加载器
    
    Args:
        vuln_type: 漏洞类型
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        batch_size: 批次大小
        max_samples: 最大样本数
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 加载数据
    loader = DatasetLoader()
    contracts = loader.load_dataset(vuln_type, max_samples)
    
    # 统计信息
    stats = loader.get_dataset_statistics(contracts)
    print(f"Dataset statistics for {vuln_type}:")
    print(f"  Total: {stats['total']}, Positive: {stats['positive']}, Negative: {stats['negative']}")
    print(f"  Positive ratio: {stats['positive_ratio']:.2%}")
    print(f"  Has inheritance: {stats['has_inheritance']}")
    print(f"  Avg nodes: {stats['avg_nodes']:.2f}, Avg edges: {stats['avg_edges']:.2f}")
    
    # 划分数据集
    np.random.seed(42)
    indices = np.random.permutation(len(contracts))
    
    train_size = int(len(indices) * train_ratio)
    val_size = int(len(indices) * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_contracts = [contracts[i] for i in train_indices]
    val_contracts = [contracts[i] for i in val_indices]
    test_contracts = [contracts[i] for i in test_indices]
    
    # 创建数据集
    train_dataset = ContractGraphDataset(train_contracts, vuln_type, max_nodes=max_nodes)
    val_dataset = ContractGraphDataset(val_contracts, vuln_type, max_nodes=max_nodes)
    test_dataset = ContractGraphDataset(test_contracts, vuln_type, max_nodes=max_nodes)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader


# 测试代码
if __name__ == "__main__":
    # 测试数据加载
    loader = DatasetLoader()
    
    print("=" * 50)
    print("Loading reentrancy dataset...")
    contracts = loader.load_dataset('reentrancy', max_samples=100)
    
    stats = loader.get_dataset_statistics(contracts)
    print(f"\nStatistics:")
    print(f"  Total: {stats['total']}")
    print(f"  Positive: {stats['positive']} ({stats['positive_ratio']:.2%})")
    print(f"  Has inheritance: {stats['has_inheritance']}")
    
    # 显示一个正样本的例子
    positive_contracts = [c for c in contracts if c.label == 1]
    if positive_contracts:
        print(f"\nPositive sample example:")
        c = positive_contracts[0]
        print(f"  Contract: {c.contract_name}")
        print(f"  File ID: {c.file_id}")
        print(f"  Functions: {c.num_functions}")
        print(f"  Inheritance depth: {c.inheritance_depth}")
        print(f"  Parent contracts: {c.parent_contracts}")
        print(f"  Has reentrancy pattern: {c.has_reentrancy_pattern}")
        print(f"  Graph nodes: {len(c.graph_nodes)}")
        print(f"  Graph edges: {len(c.graph_edges)}")
