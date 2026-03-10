
"""
带类别权重平衡的训练脚本
用于处理类别不平衡问题
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.data_loader import DualLayerGraphDataset
from main.models.dual_layer_gnn import DualLayerGNNModel
from main.training.trainer import Trainer
from main.training.loss import HybridLoss
from main.config import *


def compute_class_weights(dataset, num_classes=4):
    """
    根据数据集标签分布计算类别权重
    使用 inverse frequency 方法
    """
    labels_sum = torch.zeros(num_classes)
    total = 0
    
    for i in range(len(dataset)):
        labels = dataset[i]['labels']
        labels_sum += labels
        total += 1
    
    # 计算类别权重：使用对数平滑
    class_weights = []
    for i in range(num_classes):
        pos_count = labels_sum[i].item()
        neg_count = total - pos_count
        
        if pos_count > 0:
            # 使用平方根平滑的inverse frequency
            weight = (neg_count / pos_count) ** 0.5
            weight = min(weight, 20.0)  # 限制最大权重
        else:
            weight = 1.0
        
        class_weights.append(weight)
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    return class_weights, labels_sum


def main():
    print("=" * 60)
    print("带类别权重平衡的训练")
    print("=" * 60)
    
    # 数据目录
    data_dir = project_root / "data" / "samples"
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    test_dir = data_dir / "test"
    
    # 加载数据集
    vulnerability_types = [
        'reentrancy',
        'unchecked external call',
        'ether frozen',
        'ether strict equality'
    ]
    
    print("\n[1] Loading datasets...")
    train_dataset = DualLayerGraphDataset(str(train_dir), vulnerability_types=vulnerability_types)
    val_dataset = DualLayerGraphDataset(str(valid_dir), vulnerability_types=vulnerability_types)
    test_dataset = DualLayerGraphDataset(str(test_dir), vulnerability_types=vulnerability_types)
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Valid: {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    # 计算类别权重
    print("\n[2] Computing class weights...")
    class_weights, labels_sum = compute_class_weights(train_dataset)
    
    print("  Label distribution:")
    for i, vt in enumerate(vulnerability_types):
        print(f"    {vt}: {int(labels_sum[i].item())} / {len(train_dataset)}")
    
    print("  Class weights:")
    for i, vt in enumerate(vulnerability_types):
        print(f"    {vt}: {class_weights[i].item():.2f}")
    
    # 模型配置
    model_config = {
        'node_feature_dim': 64,
        'hidden_dim': 128,
        'num_classes': 4,
        'use_upper_graph': True,
        'gate_type': 'adaptive',
    }
    
    print("\n[3] Building model...")
    model = DualLayerGNNModel(**model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # 损失函数配置 - 使用类别权重
    loss_config = {
        'baseline_weight': 1.0,
        'pseudo_weight': 0.0,  # 关闭伪标签
        'confidence_weight': 0.0,
        'focal_gamma': 2.0,
        'use_focal': True,
        'class_weights': class_weights,
    }
    
    loss_fn = HybridLoss(**loss_config)
    
    # 训练配置
    training_config = {
        'epochs': 30,
        'pretrain_epochs': 5,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'patience': 10,
        'output_dir': str(project_root / "output"),
    }
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[4] Device: {device}")
    
    # 创建Trainer
    print("\n[5] Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        config=training_config,
        loss_fn=loss_fn,
        vulnerability_types=vulnerability_types,
        device=device,
    )
    
    # 训练
    print("\n[6] Starting training...")
    print("=" * 60)
    
    history = trainer.train(epochs=training_config['epochs'])
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    # 最终评估
    print("\n[7] Final evaluation on test set...")
    test_metrics = trainer.evaluate()
    
    print("\nTest Results:")
    print(f"  Accuracy: {test_metrics.get('accuracy', 0):.4f}")
    print(f"  Macro F1: {test_metrics.get('macro_f1', 0):.4f}")
    print(f"  Macro Precision: {test_metrics.get('macro_precision', 0):.4f}")
    print(f"  Macro Recall: {test_metrics.get('macro_recall', 0):.4f}")
    
    print(f"\nOutput: {project_root / 'output'}")


if __name__ == "__main__":
    main()

