"""
上层图构建器
根据继承关系构建上层图（继承关系图）
"""

import torch
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class InheritanceGraphData:
    """继承图数据"""
    nodes: List[str]  # 合约节点
    edges: List[Tuple[str, str, str]]  # (父合约, 子合约, 继承类型)
    node_features: torch.Tensor  # (num_nodes, feature_dim)
    edge_index: torch.Tensor  # (2, num_edges)
    edge_type: torch.Tensor  # (num_edges,)


class UpperGraphBuilder:
    """
    上层图构建器
    
    功能：
    1. 分析合约的继承关系
    2. 构建继承关系图
    3. 提取继承图特征
    """
    
    # 继承类型
    INHERITANCE_TYPES = {
        'is': 0,       # 普通继承
        'via': 1,      # 通过库继承
    }
    
    def __init__(self, max_nodes: int = 50):
        """
        Args:
            max_nodes: 最大节点数
        """
        self.max_nodes = max_nodes
    
    def extract_inheritance_from_source(self, source_code: str) -> List[str]:
        """
        从源代码中提取继承关系
        
        Args:
            source_code: Solidity源代码
            
        Returns:
            父合约列表
        """
        if not source_code:
            return []
        
        # 匹配 contract X is Y, Z
        pattern = r'contract\s+(\w+)\s+is\s+([\w,\s]+)'
        matches = re.findall(pattern, source_code)
        
        parents = []
        for match in matches:
            # match[1] 是父合约列表
            parent_list = [p.strip() for p in match[1].split(',')]
            parents.extend(parent_list)
        
        return parents
    
    def build_inheritance_graph(
        self,
        contracts: List[Dict],
    ) -> nx.DiGraph:
        """
        构建继承关系图
        
        Args:
            contracts: 合约列表，每个合约包含:
                - name: 合约名
                - parents: 父合约列表
                
        Returns:
            NetworkX有向图
        """
        G = nx.DiGraph()
        
        # 添加节点
        for contract in contracts:
            G.add_node(contract['name'])
        
        # 添加边（从父合约指向子合约，表示继承方向）
        for contract in contracts:
            for parent in contract.get('parents', []):
                if parent in G.nodes:
                    G.add_edge(parent, contract['name'], type='inherit')
        
        return G
    
    def compute_inheritance_features(
        self,
        G: nx.DiGraph,
        contract_name: str,
    ) -> Dict[str, float]:
        """
        计算单个合约的继承特征
        
        Args:
            G: 继承图
            contract_name: 合约名
            
        Returns:
            特征字典
        """
        if contract_name not in G:
            return {
                'in_degree': 0,
                'out_degree': 0,
                'depth': 0,
                'num_ancestors': 0,
                'num_descendants': 0,
                'has_inheritance': 0,
            }
        
        # 入度（子合约数量）
        in_degree = G.in_degree(contract_name)
        
        # 出度（父合约数量）
        out_degree = G.out_degree(contract_name)
        
        # 祖先数量
        ancestors = nx.ancestors(G, contract_name)
        
        # 后代数量
        descendants = nx.descendants(G, contract_name)
        
        # 计算深度（从根节点到当前节点的距离）
        try:
            # 找到所有根节点（没有父合约的节点）
            roots = [n for n in G.nodes() if G.out_degree(n) == 0]
            # 计算到最近根节点的距离
            depths = [nx.shortest_path_length(G, root, contract_name) 
                     for root in roots if nx.has_path(G, root, contract_name)]
            depth = min(depths) if depths else 0
        except:
            depth = 0
        
        return {
            'in_degree': in_degree,
            'out_degree': out_degree,
            'depth': depth,
            'num_ancestors': len(ancestors),
            'num_descendants': len(descendants),
            'has_inheritance': 1 if (in_degree > 0 or out_degree > 0) else 0,
        }
    
    def build_upper_graph_sample(
        self,
        target_contract: str,
        contracts: List[Dict],
        max_depth: int = 2,
    ) -> Optional[InheritanceGraphData]:
        """
        为目标合约构建上层图
        
        Args:
            target_contract: 目标合约名
            contracts: 所有合约列表
            max_depth: 最大继承深度
            
        Returns:
            继承图数据或None
        """
        # 构建全局继承图
        G = self.build_inheritance_graph(contracts)
        
        if target_contract not in G:
            return None
        
        # 提取目标合约的子图（指定深度）
        ancestors = nx.ancestors(G, target_contract)
        descendants = nx.descendants(G, target_contract)
        
        # 收集相关节点
        relevant_nodes = {target_contract}
        relevant_nodes.update(ancestors)
        relevant_nodes.update(descendants)
        
        # 限制节点数量
        if len(relevant_nodes) > self.max_nodes:
            # 只保留最近的节点
            relevant_nodes = {target_contract}
            for _ in range(max_depth):
                new_nodes = set()
                for node in relevant_nodes:
                    new_nodes.update(G.predecessors(node))  # 父合约
                    new_nodes.update(G.successors(node))    # 子合约
                relevant_nodes.update(new_nodes)
                if len(relevant_nodes) >= self.max_nodes:
                    break
        
        # 创建子图
        subgraph = G.subgraph(relevant_nodes).copy()
        
        # 构建节点特征
        node_list = list(subgraph.nodes())
        node_features = []
        for node in node_list:
            features = self.compute_inheritance_features(subgraph, node)
            feature_vec = [
                features['in_degree'],
                features['out_degree'],
                features['depth'],
                features['num_ancestors'],
                features['num_descendants'],
                features['has_inheritance'],
            ]
            node_features.append(feature_vec)
        
        node_features = torch.tensor(node_features, dtype=torch.float32)
        
        # 构建边索引
        edges = []
        edge_types = []
        for u, v in subgraph.edges():
            edges.append((node_list.index(u), node_list.index(v)))
            edge_types.append(self.INHERITANCE_TYPES['is'])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_type = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 1), dtype=torch.long)
            edge_type = torch.zeros((1,), dtype=torch.long)
        
        return InheritanceGraphData(
            nodes=node_list,
            edges=list(subgraph.edges()),
            node_features=node_features,
            edge_index=edge_index,
            edge_type=edge_type,
        )
    
    def should_build_upper_graph(
        self,
        inheritance_features: torch.Tensor,
        threshold: float = 0.3,
    ) -> torch.Tensor:
        """
        判断是否需要构建上层图
        
        基于继承相关性预测结果
        
        Args:
            inheritance_features: (batch_size, 8) 继承特征
            threshold: 阈值
            
        Returns:
            (batch_size,) bool tensor
        """
        # 简单判断：有继承关系就构建
        has_inheritance = inheritance_features[:, 0] > 0  # 继承深度 > 0
        return has_inheritance


class AdaptiveUpperGraphBuilder(UpperGraphBuilder):
    """
    自适应上层图构建器
    
    根据继承相关性预测结果决定是否构建上层图
    """
    
    def __init__(
        self,
        max_nodes: int = 50,
        confidence_threshold: float = 0.5,
    ):
        super().__init__(max_nodes)
        self.confidence_threshold = confidence_threshold
    
    def conditional_build(
        self,
        target_contract: str,
        contracts: List[Dict],
        inheritance_prob: float,
        confidence: float,
    ) -> Optional[InheritanceGraphData]:
        """
        条件构建上层图
        
        只有当继承相关性概率和置信度都足够高时才构建
        
        Args:
            target_contract: 目标合约
            contracts: 所有合约
            inheritance_prob: 继承相关性概率
            confidence: 预测置信度
            
        Returns:
            继承图数据或None
        """
        # 决策：只有当概率和置信度都足够高时才构建
        if inheritance_prob >= self.confidence_threshold and confidence >= 0.5:
            return self.build_upper_graph_sample(target_contract, contracts)
        return None
