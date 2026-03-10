"""
分析Dataset各漏洞类型的正负样本分布
"""

import pandas as pd
from pathlib import Path

DATASET_ROOT = Path("E:/OneDrive/muti/sc/Dataset")

# 漏洞类型列表
VULN_TYPES = [
    "dangerous delegatecall",
    "ether frozen", 
    "timestamp dependency",
    "reentrancy",
    "unchecked external call",
    "integer overflow",
    "block number dependency",
    "ether strict equality"
]

def analyze_dataset():
    """分析各漏洞类型的数据分布"""
    
    print("=" * 80)
    print("Dataset 各漏洞类型数据分布分析")
    print("=" * 80)
    
    results = []
    
    for vuln_type in VULN_TYPES:
        csv_path = DATASET_ROOT / f"{vuln_type}.csv"
        
        if not csv_path.exists():
            print(f"[WARN] {vuln_type}: CSV文件不存在")
            continue
        
        df = pd.read_csv(csv_path)
        
        total = len(df)
        positive = (df['ground truth'] == 1).sum()
        negative = (df['ground truth'] == 0).sum()
        
        if total > 0:
            pos_ratio = positive / total * 100
        else:
            pos_ratio = 0
        
        results.append({
            'vuln_type': vuln_type,
            'positive': positive,
            'negative': negative,
            'total': total,
            'pos_ratio': pos_ratio
        })
        
        print(f"\n[{vuln_type}]")
        print(f"   正样本: {positive} ({pos_ratio:.2f}%)")
        print(f"   负样本: {negative}")
        print(f"   总计:   {total}")
    
    # 按正样本比例排序
    print("\n" + "=" * 80)
    print("按正样本比例排序（适合实验的优先级）")
    print("=" * 80)
    
    results_sorted = sorted(results, key=lambda x: x['pos_ratio'], reverse=True)
    
    for i, r in enumerate(results_sorted, 1):
        # 推荐平衡比例
        if r['positive'] >= 50:
            recommended_ratio = "1:2"
            recommended_neg = r['positive'] * 2
        else:
            recommended_ratio = "1:1" 
            recommended_neg = r['positive']
        
        print(f"\n{i}. {r['vuln_type']}")
        print(f"   正样本: {r['positive']}, 负样本: {r['negative']}, 比例: {r['pos_ratio']:.2f}%")
        print(f"   平衡建议: 1:1 -> 正{r['positive']}, 负{r['positive']} (共{2*r['positive']})")
        print(f"            1:2 -> 正{r['positive']}, 负{recommended_neg} (共{r['positive']+recommended_neg})")
    
    return results

if __name__ == "__main__":
    analyze_dataset()
