"""
训练入口脚本
实现完整的两阶段训练流程
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from optimized_config import (
    MODEL_CONFIG, TRAIN_CONFIG, GATE_CONFIG, 
    UPPER_GRAPH_CONFIG, LOSS_CONFIG, VULNERABILITY_CONFIG
)
from utils.data_loader import DualLayerGraphDataset, default_collate_fn
from main.models.dual_layer_gnn import DualLayerGNNModel, MultiTaskDualLayerModel
from main.training.trainer import Trainer


def create_sample_data(output_dir: str, num_samples: int = 50,
                      imbalance_ratio: float = 0.3,
                      upper_graph_ratio: float = 0.6,
                      inheritance_vuln_correlation: float = 0.7):
    """
    创建样本训练数据用于测试
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
        imbalance_ratio: 正负样本比例（越高正样本越多）
        upper_graph_ratio: 有上层图的比例
        inheritance_vuln_correlation: 继承关系与漏洞的相关性（0-1）
            - 如果有继承关系相关漏洞，上层图概率更高
    """
    os.makedirs(output_dir, exist_ok=True)
    
    vulnerability_types = VULNERABILITY_CONFIG['types']
    
    for i in range(num_samples):
        # 生成随机节点特征 (每节点64维)
        num_nodes = np.random.randint(5, 20)
        node_features = np.random.randn(num_nodes, 64).astype(np.float32)
        
        # 生成随机边 (控制在10条以内)
        num_edges = min(num_nodes * 2, np.random.randint(3, 10))
        edge_index = []
        edge_type = []
        
        for _ in range(num_edges):
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src != dst:
                edge_index.append([src, dst])
                edge_type.append(np.random.randint(0, 3))  # 3种边类型
        
        # 生成标签 (多标签分类) - 增加正样本比例
        labels = {}
        has_inheritance_vuln = False  # 是否有继承关系相关的漏洞
        
        for vuln_type in vulnerability_types:
            # 根据漏洞类型设置不同的概率
            if vuln_type == 'reentrancy':
                # 重入漏洞是继承关系相关的，提高概率
                labels[vuln_type] = 1 if np.random.random() < imbalance_ratio else 0
                if labels[vuln_type] == 1:
                    has_inheritance_vuln = True
            elif vuln_type == 'unchecked external call':
                labels[vuln_type] = 1 if np.random.random() < imbalance_ratio * 0.8 else 0
                if labels[vuln_type] == 1:
                    has_inheritance_vuln = True
            elif vuln_type == 'ether frozen':
                labels[vuln_type] = 1 if np.random.random() < imbalance_ratio * 2.0 else 0
            else:  # ether strict equality
                labels[vuln_type] = 1 if np.random.random() < imbalance_ratio * 0.5 else 0
        
        # 决定是否有上层图 - 与漏洞相关联
        # 如果有继承关系相关漏洞，提高有上层图的概率
        if has_inheritance_vuln:
            # 有继承相关漏洞时，高概率有上层图
            has_upper = np.random.random() < upper_graph_ratio
        else:
            # 没有继承相关漏洞时，降低上层图概率
            has_upper = np.random.random() < (upper_graph_ratio * 0.5)
        
        upper_graphs = []
        suspicious_score = 0.0
        
        if has_upper:
            # 生成上层图 (继承关系图)
            num_upper_nodes = np.random.randint(2, 6)
            upper_node_features = np.random.randn(num_upper_nodes, 64).astype(np.float32)
            
            # 生成继承关系边
            upper_edges = []
            for j in range(num_upper_nodes - 1):
                upper_edges.append([j + 1, j])  # 子合约指向父合约
            
            # 计算可疑分数 - 如果有漏洞，分数更高
            if has_inheritance_vuln:
                suspicious_score = np.random.uniform(0.4, 0.9)
            else:
                suspicious_score = np.random.uniform(0.1, 0.5)
            
            upper_graphs.append({
                'node_features': upper_node_features.tolist(),
                'edge_index': upper_edges,
                'edge_type': [0] * len(upper_edges)
            })
        
        # 构建样本数据
        sample = {
            'target_contract': f'Contract_{i:04d}',
            'target_graph': {
                'node_features': node_features.tolist(),
                'edge_index': edge_index,
                'edge_type': edge_type
            },
            'upper_graphs': upper_graphs,
            'has_upper_graph': has_upper,
            'labels': labels,
            'suspicious_score': suspicious_score
        }
        
        # 保存为JSON文件
        filepath = os.path.join(output_dir, f'sample_{i:04d}.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sample, f, indent=2)
    
    print(f"[OK] Created {num_samples} sample files in {output_dir}")
    print(f"  - Positive ratio: ~{imbalance_ratio*100:.0f}%")
    print(f"  - Upper graph ratio: ~{upper_graph_ratio*100:.0f}%")
    print(f"  - Inheritance-vuln correlation: {inheritance_vuln_correlation*100:.0f}%")


def split_dataset(data_dir: str, train_ratio: float = 0.7, 
                  val_ratio: float = 0.15, test_ratio: float = 0.15):
    """
    划分训练集、验证集、测试集
    """
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    np.random.shuffle(all_files)
    
    n = len(all_files)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    splits = {
        'train': all_files[:train_end],
        'valid': all_files[train_end:val_end],
        'test': all_files[val_end:]
    }
    
    for split_name, files in splits.items():
        split_dir = os.path.join(data_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for f in files:
            src = os.path.join(data_dir, f)
            dst = os.path.join(split_dir, f)
            # 移动文件
            with open(src, 'r') as sf:
                data = json.load(sf)
            with open(dst, 'w') as df:
                json.dump(data, df, indent=2)
            os.remove(src)
    
    print(f"[OK] Dataset split: train={len(splits['train'])}, valid={len(splits['valid'])}, test={len(splits['test'])}")
    return splits


def build_model(config: dict, vulnerability_types: list):
    """
    构建模型
    """
    # 使用配置中的门控类型
    gate_type = config.get('gate_type', 'adaptive')
    
    model = DualLayerGNNModel(
        node_feature_dim=config.get('node_feature_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        num_classes=len(vulnerability_types),
        num_edge_types=8,
        use_upper_graph=config.get('use_upper_graph', True),
        gate_type=gate_type,
        upper_graph_dim=config.get('upper_graph_dim', 64)
    )
    return model


def main():
    parser = argparse.ArgumentParser(description='Smart Contract Vulnerability Detection Training')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['num_epochs'], help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'], help='Batch size')
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['learning_rate'], help='Learning rate')
    parser.add_argument('--pretrain_epochs', type=int, default=TRAIN_CONFIG['pretrain_epochs'], help='Pretrain epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--create_samples', action='store_true', help='Create sample data first')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to create')
    parser.add_argument('--imbalance_ratio', type=float, default=0.3, help='Positive sample ratio (0-1)')
    parser.add_argument('--upper_graph_ratio', type=float, default=0.6, help='Upper graph ratio (0-1)')
    parser.add_argument('--gate_type', type=str, default='adaptive', 
                       choices=['adaptive', 'always', 'never', 'simple'],
                       help='Gate type for upper graph')
    
    args = parser.parse_args()
    
    # 漏洞类型
    vulnerability_types = VULNERABILITY_CONFIG['types']
    
    # 数据目录
    if args.data_dir:
        data_dir = args.data_dir
    else:
        base_dir = Path(__file__).parent
        data_dir = base_dir / 'data' / 'samples'
    
    # 确保目录是字符串
    data_dir_str = str(data_dir)
    
    # 检查数据是否存在 - 首先检查子目录（train/test/valid）
    train_dir = os.path.join(data_dir_str, 'train')
    if os.path.exists(train_dir):
        # 使用子目录结构
        print("[INFO] Using train/valid/test split from subdirectories")
        train_dir_str = train_dir
        valid_dir_str = os.path.join(data_dir_str, 'valid')
        test_dir_str = os.path.join(data_dir_str, 'test')
        
        train_files = [f for f in os.listdir(train_dir_str) if f.endswith('.json')]
        valid_files = [f for f in os.listdir(valid_dir_str) if f.endswith('.json')]
        test_files = [f for f in os.listdir(test_dir_str) if f.endswith('.json')]
        
        print(f"  Train: {len(train_files)} samples")
        print(f"  Valid: {len(valid_files)} samples")
        print(f"  Test:  {len(test_files)} samples")
        
        # 加载数据集
        train_dataset = DualLayerGraphDataset(train_dir_str, vulnerability_types=vulnerability_types)
        val_dataset = DualLayerGraphDataset(valid_dir_str, vulnerability_types=vulnerability_types)
        test_dataset = DualLayerGraphDataset(test_dir_str, vulnerability_types=vulnerability_types)
        
    else:
        # 检查数据是否存在
        if args.create_samples:
            print("\n" + "="*60)
            print("Creating sample data...")
            print("="*60)
            create_sample_data(str(data_dir_str), args.num_samples,
                             imbalance_ratio=args.imbalance_ratio,
                             upper_graph_ratio=args.upper_graph_ratio)
        
        sample_files = [f for f in os.listdir(data_dir_str) if f.endswith('.json')]
        if len(sample_files) == 0:
            print("[ERROR] No data files found. Use --create_samples to generate sample data.")
            return
        
        print("\n" + "="*60)
        print("Loading datasets...")
        print("="*60)
        
        # 加载数据集
        full_dataset = DualLayerGraphDataset(data_dir_str, vulnerability_types=vulnerability_types)
        
        # 简单划分：前70%训练，后15%验证，最后15%测试
        n = len(full_dataset)
        train_size = int(n * 0.7)
        val_size = int(n * 0.15)
        
        # 创建子集
        class Subset(DualLayerGraphDataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
                self.vulnerability_types = dataset.vulnerability_types
                self.data_files = [dataset.data_files[i] for i in indices]
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]
        
        train_dataset = Subset(full_dataset, list(range(train_size)))
        val_dataset = Subset(full_dataset, list(range(train_size, train_size + val_size)))
        test_dataset = Subset(full_dataset, list(range(train_size + val_size, n)))
        
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Valid: {len(val_dataset)} samples")
        print(f"  Test:  {len(test_dataset)} samples")
    
    # 构建模型
    print("\n" + "="*60)
    print("Building model...")
    print("="*60)
    
    # 更新MODEL_CONFIG使用命令行指定的gate_type
    model_config = MODEL_CONFIG.copy()
    model_config['gate_type'] = args.gate_type
    
    model = build_model(model_config, vulnerability_types)
    print(f"  Model: DualLayerGNNModel")
    print(f"  Hidden dim: {MODEL_CONFIG['hidden_dim']}")
    print(f"  Num classes: {len(vulnerability_types)}")
    print(f"  Gate type: {args.gate_type}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 配置
    config = {
        **TRAIN_CONFIG,
        'num_classes': len(vulnerability_types),
        'vulnerability_types': vulnerability_types,
        'output_dir': str(Path(__file__).parent / 'output'),
        'device': args.device
    }
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['pretrain_epochs'] = args.pretrain_epochs
    config['epochs'] = args.epochs
    
    # 创建训练器
    print("\n" + "="*60)
    print("Initializing trainer...")
    print("="*60)
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        config=config,
        device=args.device
    )
    
    # 开始训练
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    history = trainer.train(epochs=args.epochs)
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Best epoch: {trainer.best_epoch + 1}")
    print(f"Best metric (macro F1): {trainer.best_metric:.4f}")
    print(f"Output directory: {trainer.output_dir}")


if __name__ == '__main__':
    main()
