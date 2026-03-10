"""
数据转换脚本
将原始 Solidity 源代码和 CSV 标签转换为模型可用的 JSON 格式
"""
import os
import json
import csv
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

# 配置
DATASET_DIR = Path(__file__).parent.parent / "Dataset"
OUTPUT_DIR = Path(__file__).parent / "data" / "samples"

# 漏洞类型（4种）
VULNERABILITY_TYPES = [
    'reentrancy',
    'unchecked external call',
    'ether frozen',
    'ether strict equality'
]

# 节点特征维度
NODE_FEATURE_DIM = 64


class SimpleSolidityParser:
    """简化版 Solidity 解析器"""
    
    CONTRACT_PATTERN = re.compile(r'^\s*contract\s+(\w+)(?:\s+is\s+([\w\s,]+))?', re.MULTILINE)
    FUNCTION_PATTERN = re.compile(r'^\s*(function|modifier)\s+(\w+)', re.MULTILINE)
    VARIABLE_PATTERN = re.compile(r'^\s*(uint|int|address|bool|string|bytes|mapping|struct|enum)[^\n]+(\w+)\s*;', re.MULTILINE)
    SEND_PATTERN = re.compile(r'\.(send|transfer|call)\s*\(', re.MULTILINE)
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.contracts = []
        self.functions = []
        self.variables = []
        self.inheritance = {}
        self.external_calls = []
        self._parse()
    
    def _parse(self):
        for match in self.CONTRACT_PATTERN.finditer(self.source_code):
            contract_name = match.group(1)
            parents = match.group(2)
            self.contracts.append(contract_name)
            self.inheritance[contract_name] = []
            if parents:
                self.inheritance[contract_name] = [p.strip() for p in parents.split(',')]
        
        for match in self.FUNCTION_PATTERN.finditer(self.source_code):
            self.functions.append({'type': match.group(1), 'name': match.group(2)})
        
        for match in self.VARIABLE_PATTERN.finditer(self.source_code):
            self.variables.append({'type': match.group(1), 'name': match.group(2)})
        
        for match in self.SEND_PATTERN.finditer(self.source_code):
            self.external_calls.append(match.group(1))


def extract_node_features(node_type: str, info: Dict) -> List[float]:
    """提取节点特征向量 (64维)"""
    features = []
    
    # 节点类型编码 (4维)
    type_map = {'contract': [1, 0, 0, 0], 'function': [0, 1, 0, 0], 'variable': [0, 0, 1, 0], 'other': [0, 0, 0, 1]}
    features.extend(type_map.get(node_type, [0, 0, 0, 1]))
    
    if node_type == 'contract':
        has_inh = 1 if info.get('has_inheritance') else 0
        num_funcs = min(info.get('num_functions', 0) / 20.0, 1.0)
        num_vars = min(info.get('num_variables', 0) / 10.0, 1.0)
        num_calls = min(info.get('num_external_calls', 0) / 10.0, 1.0)
        features.extend([has_inh, num_funcs, num_vars, num_calls])
        
        is_erc20 = 1 if 'ERC20' in info.get('name', '') else 0
        is_erc721 = 1 if 'ERC721' in info.get('name', '') else 0
        has_owner = 1 if 'Ownable' in info.get('name', '') else 0
        has_safe = 1 if 'SafeMath' in info.get('name', '') else 0
        features.extend([is_erc20, is_erc721, has_owner, has_safe])
        
    elif node_type == 'function':
        is_public = 1 if info.get('visibility') == 'public' else 0
        is_external = 1 if info.get('visibility') == 'external' else 0
        is_payable = 1 if info.get('is_payable', False) else 0
        is_view = 1 if info.get('is_view', False) else 0
        is_constructor = 1 if info.get('is_constructor', False) else 0
        has_call = 1 if info.get('has_external_call', False) else 0
        features.extend([is_public, is_external, is_payable, is_view, is_constructor, has_call])
    
    while len(features) < NODE_FEATURE_DIM:
        features.append(0.0)
    
    return features[:NODE_FEATURE_DIM]


def build_contract_graph(source_code: str, contract_name: str) -> Dict:
    """为单个合约构建图结构"""
    parser = SimpleSolidityParser(source_code)
    
    nodes = []
    edges = []
    node_id_map = {}
    node_cnt = 0
    
    # 合约节点
    contract_info = {
        'name': contract_name,
        'has_inheritance': len(parser.inheritance.get(contract_name, [])) > 0,
        'num_functions': len(parser.functions),
        'num_variables': len(parser.variables),
        'num_external_calls': len(parser.external_calls)
    }
    
    node_id_map[contract_name] = node_cnt
    nodes.append({
        'id': node_cnt,
        'type': 'contract',
        'name': contract_name,
        'features': extract_node_features('contract', contract_info)
    })
    node_cnt += 1
    
    # 函数节点
    for func in parser.functions:
        if func['type'] == 'function':
            func_name = func['name']
            node_id_map[f"{contract_name}.{func_name}"] = node_cnt
            
            func_info = {
                'visibility': 'public',
                'is_payable': False,
                'is_view': False,
                'is_constructor': func_name.lower() == contract_name.lower(),
                'has_external_call': any(c in str(parser.external_calls) for c in ['call', 'send', 'transfer'])
            }
            
            nodes.append({
                'id': node_cnt,
                'type': 'function',
                'name': func_name,
                'features': extract_node_features('function', func_info)
            })
            
            edges.append({'src': node_id_map[contract_name], 'dst': node_cnt, 'type': 'has_function'})
            node_cnt += 1
    
    # 变量节点 (限制数量)
    for var in parser.variables[:8]:
        var_name = var['name']
        node_id_map[f"{contract_name}.{var_name}"] = node_cnt
        
        nodes.append({
            'id': node_cnt,
            'type': 'variable',
            'name': var_name,
            'features': extract_node_features('variable', {})
        })
        
        edges.append({'src': node_id_map[contract_name], 'dst': node_cnt, 'type': 'has_variable'})
        node_cnt += 1
    
    # 继承边
    for parent in parser.inheritance.get(contract_name, []):
        if parent in node_id_map:
            edges.append({'src': node_id_map[contract_name], 'dst': node_id_map[parent], 'type': 'inherit'})
    
    return {'nodes': nodes, 'edges': edges}


def convert_to_tensor_format(graph: Dict) -> Dict:
    """转换为模型tensor格式"""
    nodes = graph['nodes']
    edges = graph['edges']
    
    node_features = [n['features'] for n in nodes]
    node_id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
    
    edge_src, edge_dst, edge_types = [], [], []
    edge_type_map = {'has_function': 0, 'has_variable': 1, 'inherit': 2, 'call': 3}
    
    for edge in edges:
        src_idx = node_id_to_idx.get(edge['src'])
        dst_idx = node_id_to_idx.get(edge['dst'])
        if src_idx is not None and dst_idx is not None:
            edge_src.append(src_idx)
            edge_dst.append(dst_idx)
            edge_types.append(edge_type_map.get(edge['type'], 0))
    
    return {
        'node_features': node_features,
        'edge_index': [edge_src, edge_dst],
        'edge_type': edge_types,
        'num_nodes': len(nodes)
    }


def find_solidity_file(vuln_dir: Path, file_id: int) -> Path:
    """查找.sol文件"""
    for ext in ['sol', 'SOL', '']:
        if ext:
            sol_file = vuln_dir / f"{file_id}.{ext}"
        else:
            sol_file = vuln_dir / str(file_id)
        if sol_file.exists():
            return sol_file
    return None


def process_vulnerability_type(vuln_type: str, vuln_dir: Path, csv_path: Path) -> List[Dict]:
    """处理一种漏洞类型"""
    print(f"[处理] {vuln_type}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [错误] 无法读取CSV: {e}")
        return []
    
    print(f"  - 标签数量: {len(df)}")
    
    samples = []
    processed, failed = 0, 0
    
    for _, row in df.iterrows():
        file_id = int(row['file'])
        contract_name = str(row['contract'])
        label = int(row['ground truth'])
        
        sol_file = find_solidity_file(vuln_dir, file_id)
        if sol_file is None:
            failed += 1
            continue
        
        try:
            with open(sol_file, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
            
            graph = build_contract_graph(source_code, contract_name)
            tensor_graph = convert_to_tensor_format(graph)
            
            samples.append({
                'target_graph': tensor_graph,
                'labels': {vuln_type: label},
                'contract_name': contract_name,
                'file_id': file_id
            })
            
            processed += 1
            if processed % 500 == 0:
                print(f"    已处理: {processed}")
        
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"    [警告] {sol_file.name}: {e}")
    
    print(f"  - 成功: {processed}, 失败: {failed}")
    return samples


def merge_samples(samples_by_type: Dict[str, List[Dict]]) -> List[Dict]:
    """合并不同漏洞类型的样本"""
    merged = {}
    
    for vuln_type, samples in samples_by_type.items():
        for sample in samples:
            key = (sample['contract_name'], sample['file_id'])
            
            if key not in merged:
                merged[key] = {
                    'target_graph': sample['target_graph'],
                    'labels': {},
                    'contract_name': sample['contract_name'],
                    'file_id': sample['file_id']
                }
            
            merged[key]['labels'][vuln_type] = sample['labels'][vuln_type]
    
    result = []
    for sample in merged.values():
        multi_label = {vt: sample['labels'].get(vt, 0) for vt in VULNERABILITY_TYPES}
        result.append({
            'target_graph': sample['target_graph'],
            'labels': multi_label,
            'contract_name': sample['contract_name'],
            'file_id': sample['file_id']
        })
    
    return result


def save_samples(samples: List[Dict], output_dir: Path):
    """保存样本到JSON"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    
    n = len(samples)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    splits = {
        'train': indices[:train_end],
        'valid': indices[train_end:val_end],
        'test': indices[val_end:]
    }
    
    for split_name, split_indices in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for idx in split_indices:
            sample = samples[idx]
            
            output_data = {
                'target_contract': sample['contract_name'],
                'target_graph': sample['target_graph'],
                'upper_graphs': [],
                'has_upper_graph': False,
                'labels': sample['labels'],
                'suspicious_score': 0.0
            }
            
            filename = f"{sample['contract_name']}_{sample['file_id']}.json"
            filepath = split_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
    
    print(f"[保存] 训练集: {len(splits['train'])}, 验证集: {len(splits['valid'])}, 测试集: {len(splits['test'])}")


def main():
    print("=" * 60)
    print("数据转换工具")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    samples_by_type = {}
    
    for vuln_type in VULNERABILITY_TYPES:
        # 尝试不同的CSV文件名
        possible_names = [
            f"{vuln_type}.csv",
            f"{vuln_type.replace(' ', '_')}.csv",
            f"{vuln_type.replace(' ', '-')}.csv"
        ]
        
        csv_path = None
        for name in possible_names:
            p = DATASET_DIR / name
            if p.exists():
                csv_path = p
                break
        
        if csv_path is None:
            print(f"[跳过] {vuln_type} - 未找到CSV文件")
            continue
        
        # 源码目录
        vuln_dir = DATASET_DIR / vuln_type
        if not vuln_dir.exists():
            # 尝试带下划线的目录名
            vuln_dir = DATASET_DIR / vuln_type.replace(' ', '_')
        
        if not vuln_dir.exists():
            print(f"[跳过] {vuln_type} - 未找到源码目录")
            continue
        
        samples = process_vulnerability_type(vuln_type, vuln_dir, csv_path)
        if samples:
            samples_by_type[vuln_type] = samples
    
    if not samples_by_type:
        print("[错误] 未处理任何数据")
        return
    
    # 合并并保存
    merged_samples = merge_samples(samples_by_type)
    print(f"\n[合并] 共 {len(merged_samples)} 个独立样本")
    
    save_samples(merged_samples, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("数据转换完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

