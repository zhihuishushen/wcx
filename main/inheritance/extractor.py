"""
继承关系提取器
从Solidity代码中提取继承关系
"""

import re
from typing import Dict, List, Tuple, Optional
import networkx as nx


class InheritanceExtractor:
    """
    从Solidity源代码中提取继承关系的提取器
    
    支持以下继承模式：
    - 单继承: contract A is B {}
    - 多继承: contract A is B, C, D {}
    - 接口继承: contract Token is IERC20 {}
    """
    
    # 继承关系正则表达式
    INHERITANCE_PATTERN = re.compile(
        r'^\s*contract\s+(\w+)(?:\s+is\s+([\w\s,]+))?',
        re.MULTILINE
    )
    
    # 库使用模式
    USING_PATTERN = re.compile(
        r'using\s+(\w+)\s+for\s+([\w\s,]+);'
    )
    
    def __init__(self):
        pass
    
    def extract_from_source(self, source_code: str) -> Dict[str, List[str]]:
        """
        从源代码中提取继承关系
        
        Args:
            source_code: Solidity源代码
            
        Returns:
            继承关系字典 {contract_name: [parent_contracts]}
        """
        inheritance_relations = {}
        
        # 查找所有合约定义
        matches = self.INHERITANCE_PATTERN.findall(source_code)
        
        for contract_match in matches:
            contract_name = contract_match[0]
            inherited_contracts_str = contract_match[1]
            
            if inherited_contracts_str:
                # 解析继承的合约列表
                inherited = [
                    name.strip() 
                    for name in inherited_contracts_str.split(',')
                    if name.strip()
                ]
                inheritance_relations[contract_name] = inherited
            else:
                inheritance_relations[contract_name] = []
        
        return inheritance_relations
    
    def extract_with_details(self, source_code: str) -> Dict:
        """
        提取继承关系及其详细信息
        
        Returns:
            {
                'relations': {contract: [parents]},
                'libraries': [(library, types)],
                'interfaces': [interface_names],
                'multi_inheritance': [contracts_with_multiple_parents]
            }
        """
        relations = self.extract_from_source(source_code)
        
        # 提取库使用
        libraries = self.USING_PATTERN.findall(source_code)
        
        # 找出多继承的合约
        multi_inheritance = [
            contract for contract, parents in relations.items()
            if len(parents) > 1
        ]
        
        # 简单判断接口（以I开头通常是接口）
        interfaces = [
            parent for parents in relations.values()
            for parent in parents
            if parent.startswith('I') and len(parent) > 1
        ]
        
        return {
            'relations': relations,
            'libraries': libraries,
            'interfaces': list(set(interfaces)),
            'multi_inheritance': multi_inheritance,
            'has_inheritance': any(relations.values())
        }
    
    def build_graph(self, inheritance_relations: Dict[str, List[str]]) -> nx.DiGraph:
        """
        构建继承关系图
        
        Args:
            inheritance_relations: 继承关系字典
            
        Returns:
            NetworkX有向图（contract -> parent 表示contract继承parent）
        """
        G = nx.DiGraph()
        
        for contract, inherited_contracts in inheritance_relations.items():
            # 添加节点
            G.add_node(contract)
            
            # 添加边
            for parent in inherited_contracts:
                G.add_edge(contract, parent)
        
        return G
    
    def get_contract_info(self, inheritance_relations: Dict[str, List[str]], 
                          contract_name: str) -> Dict:
        """
        获取特定合约的继承信息
        
        Args:
            inheritance_relations: 继承关系字典
            contract_name: 合约名称
            
        Returns:
            {
                'parents': [父合约列表],
                'children': [子合约列表],
                'depth': 继承深度
            }
        """
        parents = inheritance_relations.get(contract_name, [])
        
        # 查找子合约
        children = [
            contract for contract, parents_list in inheritance_relations.items()
            if contract_name in parents_list
        ]
        
        # 计算继承深度
        depth = len(parents)
        
        return {
            'parents': parents,
            'children': children,
            'depth': depth,
            'has_inheritance': len(parents) > 0 or len(children) > 0
        }
    
    def compute_inheritance_features(self, inheritance_relations: Dict[str, List[str]],
                                   contract_name: str) -> Dict[str, float]:
        """
        计算合约的继承特征
        
        Args:
            inheritance_relations: 继承关系字典
            contract_name: 合约名称
            
        Returns:
            继承特征字典
        """
        info = self.get_contract_info(inheritance_relations, contract_name)
        
        # 基本特征
        num_parents = len(info['parents'])
        num_children = len(info['children'])
        inheritance_depth = info['depth']
        
        # 继承复杂度
        complexity = (
            1.0 +  # 基础复杂度
            0.3 * num_parents +  # 父合约数量
            0.2 * num_children +  # 子合约数量
            0.1 * inheritance_depth  # 继承深度
        )
        
        # 继承影响力（有多少合约依赖这个合约）
        influence = num_children / max(1, len(inheritance_relations))
        
        return {
            'num_parents': float(num_parents),
            'num_children': float(num_children),
            'inheritance_depth': float(inheritance_depth),
            'complexity': min(complexity, 2.0),
            'influence': min(influence, 1.0),
            'has_inheritance': float(info['has_inheritance'])
        }

