"""
增强版可疑分数计算器（V3版本）
计算继承关系带来的风险可疑分数
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional


class EnhancedRiskScorer:
    """
    增强版可疑分数计算器
    
    改进点：
    1. 多维度评分（距离、权重、路径数、复杂度）
    2. 漏洞类型敏感的衰减参数
    3. 继承深度影响因子
    4. 支持V1/V2/V3多种计算公式
    """
    
    # 漏洞类型特定的衰减参数（V3版本）
    DECAY_PARAMS = {
        'reentrancy': {
            'decay': 0.4,      # 快速衰减
            'severity': 1.0,    # 高严重性
            'connection_weight': {'inherit': 0.6, 'call': 1.0, 'import': 0.5}
        },
        'unchecked external call': {
            'decay': 0.3,      # 中等衰减
            'severity': 0.9,   # 较高严重性
            'connection_weight': {'inherit': 0.35, 'call': 1.5, 'import': 0.3}
        },
        'ether frozen': {
            'decay': 0.6,      # 慢速衰减
            'severity': 0.7,   # 中等严重性
            'connection_weight': {'inherit': 1.0, 'call': 0.7, 'import': 0.5}
        },
        'ether strict equality': {
            'decay': 0.8,      # 很慢衰减
            'severity': 0.5,   # 较低严重性
            'connection_weight': {'inherit': 1.0, 'call': 0.7, 'import': 0.5}
        },
    }
    
    # 默认衰减参数
    DEFAULT_DECAY_PARAMS = {
        'decay': 0.5,
        'severity': 1.0,
        'connection_weight': {'inherit': 1.0, 'call': 0.7, 'import': 0.5}
    }
    
    def __init__(self, version: str = 'v3'):
        """
        Args:
            version: 可疑分数公式版本 ('v1', 'v2', 'v3')
        """
        self.version = version
    
    def compute_score_v1(self, risk_contract: str, 
                         inheritance_graph: nx.DiGraph,
                         vul_nodes: Set[str],
                         vuln_type: str,
                         connection_type: str = 'inherit') -> float:
        """
        V1版本：原始可疑分数计算
        
        公式：
        score = base_score * weight * connection_weight * connection_strength * (1 + 0.1 * path_count)
        """
        params = self.DECAY_PARAMS.get(vuln_type, self.DEFAULT_DECAY_PARAMS)
        
        # 找到最近的有漏洞合约
        min_dist, paths = self._find_nearest_vuln(
            risk_contract, vul_nodes, inheritance_graph
        )
        
        if min_dist == float('inf'):
            return 0.0
        
        # 基础分数（线性衰减）
        base_score = 1.0 / (min_dist + 1)
        
        # 连接强度
        connection_strength = sum(1.0 / (d + 1) for d in paths)
        
        # 路径数量
        path_count = len(paths)
        
        # 连接类型权重
        connection_weight = params['connection_weight'].get(
            connection_type, 1.0
        )
        
        # 综合分数
        score = (
            base_score * 
            params['severity'] * 
            connection_weight * 
            connection_strength * 
            (1 + 0.1 * path_count)
        )
        
        return min(score, 1.0)
    
    def compute_score_v2(self, risk_contract: str,
                        inheritance_graph: nx.DiGraph,
                        vul_nodes: Set[str],
                        vuln_type: str,
                        connection_type: str = 'inherit') -> float:
        """
        V2版本：指数衰减版本
        
        公式：
        score = base_score * weight * connection_weight * total_strength * (1 + 0.2 * path_bonus) * complexity
        """
        params = self.DECAY_PARAMS.get(vuln_type, self.DEFAULT_DECAY_PARAMS)
        
        # 找到最近的有漏洞合约
        min_dist, paths = self._find_nearest_vuln(
            risk_contract, vul_nodes, inheritance_graph
        )
        
        if min_dist == float('inf'):
            return 0.0
        
        # 基础分数（指数衰减）
        decay_rate = params['decay']
        base_score = np.exp(-decay_rate * min_dist)
        
        # 总连接强度（指数衰减）
        total_strength = sum(np.exp(-decay_rate * d) for d in paths)
        
        # 路径奖励（对数增长）
        path_count = len(paths)
        path_bonus = np.log(1 + path_count)
        
        # 复杂度因子
        complexity = self._compute_complexity_factor(
            risk_contract, inheritance_graph
        )
        
        # 连接类型权重
        connection_weight = params['connection_weight'].get(
            connection_type, 1.0
        )
        
        # 综合分数
        score = (
            base_score * 
            params['severity'] * 
            connection_weight * 
            total_strength * 
            (1 + 0.2 * path_bonus) * 
            complexity
        )
        
        return min(score, 1.0)
    
    def compute_score_v3(self, risk_contract: str,
                        inheritance_graph: nx.DiGraph,
                        vul_nodes: Set[str],
                        vuln_type: str,
                        connection_type: str = 'inherit') -> float:
        """
        V3版本：增强版可疑分数计算
        
        改进点：
        1. 双向路径查找
        2. 更细粒度的衰减参数
        3. 继承深度影响因子
        
        公式：
        score = base_score * severity * weight * path_bonus * complexity * depth_factor
        """
        params = self.DECAY_PARAMS.get(vuln_type, self.DEFAULT_DECAY_PARAMS)
        
        # 双向查找路径
        paths = self._find_all_paths(
            risk_contract, vul_nodes, inheritance_graph
        )
        
        if not paths:
            return 0.0
        
        # 找到最近的有漏洞合约
        min_dist = min(paths)
        
        # 基础分数（指数衰减）
        decay_rate = params['decay']
        base_score = np.exp(-decay_rate * min_dist)
        
        # 路径奖励（对数增长）
        path_count = len(paths)
        path_bonus = np.log(1 + path_count)
        
        # 复杂度因子
        complexity = self._compute_complexity_factor(
            risk_contract, inheritance_graph
        )
        
        # 继承深度因子
        depth_factor = self._compute_depth_factor(
            risk_contract, inheritance_graph, paths
        )
        
        # 连接类型权重
        connection_weight = params['connection_weight'].get(
            connection_type, 1.0
        )
        
        # 综合分数
        score = (
            base_score * 
            params['severity'] * 
            connection_weight * 
            (1 + 0.2 * path_bonus) * 
            complexity * 
            depth_factor
        )
        
        return min(score, 1.0)
    
    def compute_score(self, risk_contract: str,
                      inheritance_graph: nx.DiGraph,
                      vul_nodes: Set[str],
                      vuln_type: str,
                      connection_type: str = 'inherit') -> float:
        """
        计算可疑分数（根据版本选择公式）
        """
        if self.version == 'v1':
            return self.compute_score_v1(
                risk_contract, inheritance_graph, vul_nodes, 
                vuln_type, connection_type
            )
        elif self.version == 'v2':
            return self.compute_score_v2(
                risk_contract, inheritance_graph, vul_nodes,
                vuln_type, connection_type
            )
        elif self.version == 'v3':
            return self.compute_score_v3(
                risk_contract, inheritance_graph, vul_nodes,
                vuln_type, connection_type
            )
        else:
            raise ValueError(f"Unknown version: {self.version}")
    
    def compute_scores_all_types(self, risk_contract: str,
                                 inheritance_graph: nx.DiGraph,
                                 vul_nodes: Set[str],
                                 vuln_types: List[str],
                                 contract_labels: Dict) -> Dict[str, float]:
        """
        计算所有漏洞类型的可疑分数
        
        Args:
            risk_contract: 风险合约名称
            inheritance_graph: 继承关系图
            vul_nodes: 有漏洞的节点集合
            vuln_types: 漏洞类型列表
            contract_labels: 合约标签字典
            
        Returns:
            {vuln_type: score}
        """
        scores = {}
        
        # 筛选有该漏洞类型的节点
        for vuln_type in vuln_types:
            vul_nodes_with_type = {
                node for node in vul_nodes
                if contract_labels.get(node, {}).get(vuln_type, 0) == 1
            }
            
            if vul_nodes_with_type:
                scores[vuln_type] = self.compute_score(
                    risk_contract, inheritance_graph,
                    vul_nodes_with_type, vuln_type
                )
            else:
                scores[vuln_type] = 0.0
        
        return scores
    
    def _find_nearest_vuln(self, risk_contract: str,
                           vul_nodes: Set[str],
                           G: nx.DiGraph) -> Tuple[float, List[int]]:
        """
        找到最近的有漏洞合约的距离
        
        Returns:
            (最小距离, 所有路径距离列表)
        """
        paths = []
        
        for vul_node in vul_nodes:
            try:
                # 双向查找
                if nx.has_path(G, risk_contract, vul_node):
                    dist = nx.shortest_path_length(G, risk_contract, vul_node)
                    paths.append(dist)
                elif nx.has_path(G, vul_node, risk_contract):
                    dist = nx.shortest_path_length(G, vul_node, risk_contract)
                    paths.append(dist)
            except nx.NetworkXNoPath:
                continue
        
        if not paths:
            return float('inf'), []
        
        return min(paths), paths
    
    def _find_all_paths(self, risk_contract: str,
                        vul_nodes: Set[str],
                        G: nx.DiGraph) -> List[int]:
        """
        找到所有到有漏洞合约的路径距离
        
        Returns:
            距离列表
        """
        paths = []
        
        for vul_node in vul_nodes:
            try:
                # 双向查找
                if nx.has_path(G, risk_contract, vul_node):
                    dist = nx.shortest_path_length(G, risk_contract, vul_node)
                    paths.append(dist)
                elif nx.has_path(G, vul_node, risk_contract):
                    dist = nx.shortest_path_length(G, vul_node, risk_contract)
                    paths.append(dist)
            except nx.NetworkXNoPath:
                continue
        
        return paths
    
    def _compute_complexity_factor(self, contract_name: str,
                                   G: nx.DiGraph) -> float:
        """
        计算合约复杂度因子
        
        基于图结构的复杂度指标：
        1. 继承深度（入度 + 出度）
        2. 连通性（与多少个节点有路径连接）
        """
        try:
            # 1. 继承深度
            in_degree = G.in_degree(contract_name) if G.has_node(contract_name) else 0
            out_degree = G.out_degree(contract_name) if G.has_node(contract_name) else 0
            degree_complexity = (in_degree + out_degree) / 10.0
            
            # 2. 连通性
            if G.has_node(contract_name):
                connectivity = sum(
                    1 for node in G.nodes()
                    if node != contract_name and (
                        nx.has_path(G, contract_name, node) or 
                        nx.has_path(G, node, contract_name)
                    )
                )
                connectivity_factor = connectivity / max(1, len(G.nodes()) - 1)
            else:
                connectivity_factor = 0.0
            
            # 3. 综合复杂度因子
            complexity_factor = 1.0 + 0.3 * degree_complexity + 0.2 * connectivity_factor
            
            return min(2.0, max(1.0, complexity_factor))
            
        except Exception:
            return 1.0
    
    def _compute_depth_factor(self, contract_name: str,
                             G: nx.DiGraph,
                             paths: List[int]) -> float:
        """
        计算继承深度因子
        
        考虑：
        1. 到最近漏洞合约的距离
        2. 合约本身的继承深度
        """
        try:
            # 1. 距离因子
            min_dist = min(paths) if paths else 1
            distance_factor = 1.0 / (1 + 0.2 * min_dist)
            
            # 2. 继承深度
            if G.has_node(contract_name):
                inheritance_depth = G.out_degree(contract_name)  # 继承的父合约数
            else:
                inheritance_depth = 0
            
            depth_factor = 1.0 + 0.1 * inheritance_depth
            
            return distance_factor * depth_factor
            
        except Exception:
            return 1.0
    
    def should_build_upper_graph(self, suspicious_score: float,
                                 threshold: float = 0.3) -> bool:
        """
        判断是否需要构建上层图
        
        Args:
            suspicious_score: 可疑分数
            threshold: 阈值
            
        Returns:
            True表示需要构建上层图
        """
        return suspicious_score >= threshold

