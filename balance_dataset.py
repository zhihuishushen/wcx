"""
数据集平衡处理模块
通过欠采样减少负样本数量，使正负样本比例接近平衡
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import random

DATASET_ROOT = Path("E:/OneDrive/muti/sc/Dataset")

# 漏洞类型及其与继承关系的关联度
VULN_INHERITANCE_RELATION = {
    "dangerous delegatecall": 3,  # 高关联：库合约、delegatecall模式
    "reentrancy": 2,              # 中关联：跨合约调用
    "integer overflow": 2,         # 中关联：库合约常用SafeMath
    "ether frozen": 1,             # 低关联
    "timestamp dependency": 1,     # 低关联
    "unchecked external call": 1,  # 低关联
    "block number dependency": 1,  # 低关联
    "ether strict equality": 1,    # 低关联
}

def load_vulnerability_data(vuln_type: str) -> pd.DataFrame:
    """
    加载特定漏洞类型的原始数据
    
    Args:
        vuln_type: 漏洞类型名称
    
    Returns:
        DataFrame with columns: file, contract, ground truth
    """
    csv_path = DATASET_ROOT / f"{vuln_type}.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df

def undersample_data(df: pd.DataFrame, ratio: float = 1.0, random_state: int = 42) -> pd.DataFrame:
    """
    欠采样：随机减少负样本数量
    
    Args:
        df: 原始DataFrame
        ratio: 目标正负样本比例 (1.0 = 1:1, 2.0 = 1:2)
        random_state: 随机种子
    
    Returns:
        平衡后的DataFrame
    """
    random.seed(random_state)
    np.random.seed(random_state)
    
    # 分离正负样本
    positive_samples = df[df['ground truth'] == 1].copy()
    negative_samples = df[df['ground truth'] == 0].copy()
    
    num_positive = len(positive_samples)
    num_negative = len(negative_samples)
    
    # 计算目标负样本数量
    target_negative = int(num_positive * ratio)
    
    if target_negative >= num_negative:
        # 不需要欠采样
        print(f"  [INFO] 负样本充足，不需要欠采样")
        return df
    
    # 随机欠采样负样本
    sampled_negative = negative_samples.sample(n=target_negative, random_state=random_state)
    
    # 合并
    balanced_df = pd.concat([positive_samples, sampled_negative], ignore_index=True)
    
    # 打乱顺序
    balanced_df = balanced_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    return balanced_df

def balance_dataset(
    vuln_type: str, 
    ratio: float = 1.0,
    random_state: int = 42,
    output_dir: Path = None
) -> pd.DataFrame:
    """
    对指定漏洞类型进行平衡处理
    
    Args:
        vuln_type: 漏洞类型
        ratio: 正负样本比例 (1.0 = 1:1, 2.0 = 1:2)
        random_state: 随机种子
        output_dir: 输出目录（可选）
    
    Returns:
        平衡后的DataFrame
    """
    print(f"\n{'='*60}")
    print(f"平衡处理: {vuln_type}")
    print(f"{'='*60}")
    
    # 加载原始数据
    df = load_vulnerability_data(vuln_type)
    
    num_pos = (df['ground truth'] == 1).sum()
    num_neg = (df['ground truth'] == 0).sum()
    
    print(f"  原始数据: 正样本={num_pos}, 负样本={num_neg}, 比例={num_pos/len(df)*100:.2f}%")
    
    # 欠采样
    balanced_df = undersample_data(df, ratio=ratio, random_state=random_state)
    
    num_pos_bal = (balanced_df['ground truth'] == 1).sum()
    num_neg_bal = (balanced_df['ground truth'] == 0).sum()
    
    print(f"  平衡后:   正样本={num_pos_bal}, 负样本={num_neg_bal}, 比例={num_pos_bal/len(balanced_df)*100:.2f}%")
    print(f"  数据量:   {len(df)} -> {len(balanced_df)} (减少 {len(df)-len(balanced_df)} 条)")
    
    # 保存平衡后的数据（可选）
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{vuln_type}_balanced.csv"
        balanced_df.to_csv(output_path, index=False)
        print(f"  保存到: {output_path}")
    
    return balanced_df

def get_balanced_datasets(
    vuln_types: List[str] = None,
    ratios: Dict[str, float] = None,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    获取多个平衡后的数据集
    
    Args:
        vuln_types: 漏洞类型列表（默认为推荐的所有类型）
        ratios: 各漏洞类型的目标比例
        random_state: 随机种子
    
    Returns:
        字典，key为漏洞类型，value为平衡后的DataFrame
    """
    if vuln_types is None:
        # 按推荐优先级排序
        vuln_types = [
            "dangerous delegatecall",  # 高关联：库合约、delegatecall
            "ether frozen",              # 数据量适中
            "timestamp dependency",     # 数据量大
            "reentrancy",                # 极端不平衡，但有代表性
        ]
    
    if ratios is None:
        # 默认全部平衡到1:1
        ratios = {vt: 1.0 for vt in vuln_types}
    
    results = {}
    
    for vuln_type in vuln_types:
        ratio = ratios.get(vuln_type, 1.0)
        balanced_df = balance_dataset(vuln_type, ratio=ratio, random_state=random_state)
        results[vuln_type] = balanced_df
    
    return results

def print_balance_summary():
    """打印平衡处理总结"""
    print("\n" + "=" * 70)
    print("推荐实验数据集（按继承关联度排序）")
    print("=" * 70)
    
    # 按推荐优先级排序
    recommended = [
        ("dangerous delegatecall", "高关联（库合约、delegatecall模式）", 75, 326, 18.70),
        ("ether frozen", "中关联（合约状态管理）", 84, 324, 20.59),
        ("timestamp dependency", "低关联（时间戳依赖）", 314, 2524, 11.06),
        ("reentrancy", "中关联（跨合约调用）", 119, 3976, 2.91),
    ]
    
    for i, (vuln, relation, pos, neg, ratio) in enumerate(recommended, 1):
        balanced_1_1 = min(pos, pos)
        balanced_1_2 = min(pos, pos * 2)
        
        print(f"\n{i}. {vuln}")
        print(f"   继承关联: {relation}")
        print(f"   原始: 正{pos}, 负{neg} ({ratio:.1f}%)")
        print(f"   平衡1:1: 正{balanced_1_1}, 负{balanced_1_1} (共{balanced_1_1*2})")
        print(f"   平衡1:2: 正{balanced_1_1}, 负{balanced_1_2} (共{balanced_1_1+balanced_1_2})")

if __name__ == "__main__":
    print_balance_summary()
