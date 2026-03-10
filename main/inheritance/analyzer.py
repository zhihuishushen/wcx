"""
继承关系分析器
整合继承关系提取和风险评估
"""

import json
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

from .extractor import InheritanceExtractor
from .risk_scorer import EnhancedRiskScorer


class InheritanceAnalyzer:
    """
    继承关系分析器
    
    功能：
    1. 从合约集合中提取和分析继承关系
    2. 识别风险节点
    3. 计算可疑分数
    4. 生成风险报告
    """
    
    # 漏洞类型分类（A/B/C类）
    VULNERABILITY_CLASSES = {
        'A_class': [
            'reentrancy',
            'unchecked external call',
            'ether frozen',
            'ether strict equality'
        ],
        'B_class': [],  # 暂时为空
        'C_class': []   # 暂时为空
    }
    
    def __init__(self, vulnerability_types: List[str] = None, 
                 version: str = 'v3'):
        """
        Args:
            vulnerability_types: 漏洞类型列表
            version: 可疑分数计算版本
        """
        self.vulnerability_types = vulnerability_types or [
            'reentrancy',
            'unchecked external call',
            'ether frozen',
            'ether strict equality'
        ]
        
        self.extractor = InheritanceExtractor()
        self.risk_scorer = EnhancedRiskScorer(version=version)
    
    def build_global_inheritance_graph(self, 
                                     contract_sources: Dict[str, str],
                                     contract_labels: Dict[str, Dict]
                                     ) -> Tuple[nx.DiGraph, Dict]:
        """
        从所有合约源代码构建全局继承关系图
        
        Args:
            contract_sources: {contract_name: source_code}
            contract_labels: {contract_name: {vuln_type: 0/1}}
            
        Returns:
            (继承关系图, 标签字典)
        """
        # 收集所有继承关系
        all_relations = {}
        
        for contract_name, source_code in contract_sources.items():
            relations = self.extractor.extract_from_source(source_code)
            for name, parents in relations.items():
                all_relations[name] = parents
        
        # 构建图
        graph = self.extractor.build_graph(all_relations)
        
        # 构建标签字典
        labels = {}
        for node in graph.nodes():
            if node in contract_labels:
                labels[node] = contract_labels[node]
            else:
                # 默认无漏洞
                labels[node] = {vuln_type: 0 for vuln_type in self.vulnerability_types}
        
        return graph, labels
    
    def classify_nodes(self, G: nx.DiGraph,
                      contract_labels: Dict) -> Tuple[Set[str], Set[str]]:
        """
        分类节点为vul_nodes和no_vul_nodes
        
        Returns:
            (有漏洞节点集合, 无漏洞节点集合)
        """
        vul_nodes = set()
        no_vul_nodes = set()
        
        for node in G.nodes():
            if node in contract_labels:
                vulnerabilities = contract_labels[node]
                if any(vulnerabilities.get(vuln, 0) == 1 for vuln in self.vulnerability_types):
                    vul_nodes.add(node)
                else:
                    no_vul_nodes.add(node)
            else:
                no_vul_nodes.add(node)
        
        return vul_nodes, no_vul_nodes
    
    def classify_risk_nodes(self, G: nx.DiGraph,
                          contract_labels: Dict) -> Tuple[Set[str], Set[str]]:
        """
        识别风险节点
        
        根据漏洞类型分类进行风险传播：
        - A类：沿入度方向（子合约方向）传播
        - B类：沿出度方向（父合约方向）传播
        - C类：双向传播
        """
        vul_nodes, no_vul_nodes = self.classify_nodes(G, contract_labels)
        risk_nodes = set()
        
        for vuln_type in self.vulnerability_types:
            # 获取有该漏洞类型的节点
            vul_nodes_with_type = {
                node for node in vul_nodes
                if contract_labels.get(node, {}).get(vuln_type, 0) == 1
            }
            
            # 根据类型确定传播方向
            if vuln_type in self.VULNERABILITY_CLASSES['A_class']:
                direction = "in"  # 子合约方向
            elif vuln_type in self.VULNERABILITY_CLASSES['B_class']:
                direction = "out"  # 父合约方向
            else:
                direction = "both"  # 双向
            
            # BFS传播
            for start_node in vul_nodes_with_type:
                risk_nodes = self._bfs_mark_risk(
                    G, start_node, risk_nodes, no_vul_nodes, direction
                )
        
        return risk_nodes, vul_nodes
    
    def _bfs_mark_risk(self, G: nx.DiGraph,
                      start_node: str,
                      risk_nodes: Set[str],
                      no_vul_nodes: Set[str],
                      direction: str) -> Set[str]:
        """
        BFS标记风险节点
        """
        visited = set()
        queue = [start_node]
        
        while queue:
            current = queue.pop(0)
            
            if current not in visited:
                visited.add(current)
                
                if current in no_vul_nodes:
                    risk_nodes.add(current)
                
                # 根据方向遍历邻居
                if direction in ["both", "out"]:
                    for neighbor in G.neighbors(current):
                        if neighbor not in visited:
                            queue.append(neighbor)
                
                if direction in ["both", "in"]:
                    for neighbor in G.predecessors(current):
                        if neighbor not in visited:
                            queue.append(neighbor)
        
        return risk_nodes
    
    def compute_suspicious_scores(self, G: nx.DiGraph,
                                  contract_labels: Dict,
                                  risk_nodes: Set[str] = None) -> Dict[str, Dict[str, float]]:
        """
        计算所有风险节点的可疑分数
        
        Returns:
            {contract_name: {vuln_type: score}}
        """
        vul_nodes, _ = self.classify_nodes(G, contract_labels)
        
        if risk_nodes is None:
            risk_nodes, _ = self.classify_risk_nodes(G, contract_labels)
        
        scores = {}
        
        for contract in risk_nodes:
            contract_scores = self.risk_scorer.compute_scores_all_types(
                contract, G, vul_nodes, self.vulnerability_types, contract_labels
            )
            scores[contract] = contract_scores
        
        return scores
    
    def generate_risk_report(self, G: nx.DiGraph,
                            contract_labels: Dict,
                            threshold: float = 0.3) -> Dict:
        """
        生成风险报告
        
        Returns:
            {
                'total_contracts': int,
                'vul_contracts': int,
                'risk_contracts': int,
                'high_risk_contracts': list,
                'per_type_stats': dict
            }
        """
        vul_nodes, no_vul_nodes = self.classify_nodes(G, contract_labels)
        risk_nodes, _ = self.classify_risk_nodes(G, contract_labels)
        
        # 计算可疑分数
        scores = self.compute_suspicious_scores(G, contract_labels, risk_nodes)
        
        # 统计高风险合约
        high_risk_contracts = []
        for contract, contract_scores in scores.items():
            max_score = max(contract_scores.values())
            if max_score >= threshold:
                high_risk_contracts.append({
                    'contract': contract,
                    'max_score': max_score,
                    'scores': contract_scores
                })
        
        # 按分数排序
        high_risk_contracts.sort(key=lambda x: x['max_score'], reverse=True)
        
        # 按漏洞类型统计
        per_type_stats = {}
        for vuln_type in self.vulnerability_types:
            type_scores = [
                scores[contract][vuln_type]
                for contract in risk_nodes
                if contract in scores
            ]
            
            if type_scores:
                per_type_stats[vuln_type] = {
                    'count': sum(1 for s in type_scores if s >= threshold),
                    'avg_score': sum(type_scores) / len(type_scores),
                    'max_score': max(type_scores)
                }
            else:
                per_type_stats[vuln_type] = {
                    'count': 0,
                    'avg_score': 0.0,
                    'max_score': 0.0
                }
        
        return {
            'total_contracts': len(G.nodes()),
            'vul_contracts': len(vul_nodes),
            'risk_contracts': len(risk_nodes),
            'high_risk_contracts': high_risk_contracts,
            'per_type_stats': per_type_stats,
            'inheritance_graph_info': {
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges()
            }
        }
    
    def save_inheritance_graph(self, G: nx.DiGraph, output_path: str):
        """
        保存继承关系图
        """
        # 转换为边列表格式
        edges = []
        for u, v in G.edges():
            edges.append({
                'from': u,
                'to': v
            })
        
        graph_data = {
            'nodes': list(G.nodes()),
            'edges': edges
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    def load_inheritance_graph(self, input_path: str) -> nx.DiGraph:
        """
        加载继承关系图
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        G = nx.DiGraph()
        
        for node in graph_data.get('nodes', []):
            G.add_node(node)
        
        for edge in graph_data.get('edges', []):
            G.add_edge(edge['from'], edge['to'])
        
        return G

