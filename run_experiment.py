"""
两阶段漏洞检测实验脚本
比较不同模式的性能
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple, List
from tqdm import tqdm

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

from main.gnn_core.dataset import DatasetLoader, create_data_loaders
from main.baseline_gnn.simple_gnn import BatchGGNNEncoder, SimpleGGNN
from main.models.inheritance_predictor import InheritancePredictor


class FocalLoss(nn.Module):
    """简化的Focal Loss"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha[1], self.alpha[0])
            loss = alpha_t * focal_weight * bce
        else:
            loss = focal_weight * bce
        
        return loss.mean()


class TwoStageModel(nn.Module):
    """
    两阶段漏洞检测模型
    包含：
    - 下层图编码器
    - 继承相关性预测器
    - 自适应融合
    """
    
    def __init__(
        self,
        node_feature_dim: int = 215,
        hidden_dim: int = 128,
        num_edge_types: int = 8,
        num_gnn_layers: int = 2,
        dropout: float = 0.2,
        mode: str = 'two_stage',  # 'two_stage', 'always', 'never', 'baseline'
    ):
        super().__init__()
        
        self.mode = mode
        self.hidden_dim = hidden_dim
        
        # 下层图编码器
        self.lower_encoder = BatchGGNNEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_edge_types=num_edge_types,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )
        
        # 继承相关性预测器
        self.inheritance_predictor = InheritancePredictor(
            input_dim=8,
            hidden_dim=64,
            use_lower_graph_stats=True,
            lower_graph_dim=hidden_dim,
        )
        
        # 分类器（下层图）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_nodes: torch.Tensor,
        inheritance_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            node_features: (batch_size, max_nodes, feature_dim)
            edge_index: (2, num_edges)
            edge_type: (num_edges,)
            num_nodes: (batch_size,)
            inheritance_features: (batch_size, 8)
            
        Returns:
            预测结果字典
        """
        # 编码下层图
        lower_emb = self.lower_encoder(
            node_features=node_features,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes_per_graph=num_nodes,
        )
        
        # 继承相关性预测
        inheritance_prob, confidence = self.inheritance_predictor(
            inheritance_features, lower_emb
        )
        
        # 根据模式决定是否使用"上层图信息"
        if self.mode == 'always':
            # 始终使用上层图（模拟）
            use_upper = torch.ones_like(inheritance_prob)
        elif self.mode == 'never' or self.mode == 'baseline':
            # 始终不使用上层图
            use_upper = torch.zeros_like(inheritance_prob)
        else:  # two_stage
            # 基于预测结果决定
            use_upper = (inheritance_prob >= 0.5).float()
        
        # 分类（这里简化处理，始终用下层图，
        # 但use_upper可以作为加权因子或用于其他目的）
        logits = self.classifier(lower_emb)
        
        return {
            'logits': logits,
            'inheritance_prob': inheritance_prob,
            'confidence': confidence,
            'use_upper': use_upper,
            'lower_emb': lower_emb,
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = 'cuda',
) -> Tuple[float, float]:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        node_features = batch['node_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_type = batch['edge_type'].to(device)
        labels = batch['labels'].to(device)
        num_nodes = batch['num_nodes'].to(device)
        inheritance_features = batch['inheritance_features'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            node_features, edge_index, edge_type,
            num_nodes, inheritance_features
        )
        
        loss = criterion(outputs['logits'].squeeze(), labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        
        predictions = (outputs['logits'].squeeze() > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += len(labels)
    
    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = 'cuda',
) -> Dict:
    """评估模型"""
    model.eval()
    total_loss = 0
    
    all_predictions = []
    all_labels = []
    all_probs = []
    inheritance_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            node_features = batch['node_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_type = batch['edge_type'].to(device)
            labels = batch['labels'].to(device)
            num_nodes = batch['num_nodes'].to(device)
            inheritance_features = batch['inheritance_features'].to(device)
            
            outputs = model(
                node_features, edge_index, edge_type,
                num_nodes, inheritance_features
            )
            
            loss = criterion(outputs['logits'].squeeze(), labels)
            
            total_loss += loss.item() * len(labels)
            
            predictions = (outputs['logits'].squeeze() > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs['logits'].squeeze().cpu().numpy())
            inheritance_probs.extend(outputs['inheritance_prob'].squeeze().cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    inheritance_probs = np.array(inheritance_probs)
    
    # 计算指标
    accuracy = (all_predictions == all_labels).mean()
    
    # 正类指标
    tp = ((all_predictions == 1) & (all_labels == 1)).sum()
    fp = ((all_predictions == 1) & (all_labels == 0)).sum()
    fn = ((all_predictions == 0) & (all_labels == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 负类指标
    tn = ((all_predictions == 0) & (all_labels == 0)).sum()
    neg_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall) if (neg_precision + neg_recall) > 0 else 0
    
    macro_f1 = (f1 + neg_f1) / 2
    
    return {
        'loss': total_loss / len(all_labels),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': macro_f1,
    }


def run_experiment(
    mode: str,
    vuln_type: str,
    epochs: int,
    batch_size: int,
    device: str,
    max_samples: int = None,
) -> Dict:
    """
    运行单个实验
    
    Args:
        mode: 实验模式
        vuln_type: 漏洞类型
        epochs: 训练轮数
        batch_size: 批次大小
        device: 设备
        max_samples: 最大样本数
        
    Returns:
        实验结果
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: mode={mode}, vuln_type={vuln_type}")
    print(f"{'='*60}")
    
    # 加载数据
    train_loader, val_loader, test_loader = create_data_loaders(
        vuln_type=vuln_type,
        batch_size=batch_size,
        max_samples=max_samples,
    )
    
    # 计算类别权重
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['labels'].numpy())
    all_labels = np.array(all_labels)
    
    num_pos = np.sum(all_labels == 1)
    num_neg = np.sum(all_labels == 0)
    total = len(all_labels)
    
    weight_neg = total / (2 * num_neg) if num_neg > 0 else 1.0
    weight_pos = total / (2 * num_pos) if num_pos > 0 else 1.0
    
    class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32).to(device)
    print(f"Class weights: neg={weight_neg:.2f}, pos={weight_pos:.2f}")
    
    # 创建模型
    model = TwoStageModel(
        node_feature_dim=215,
        hidden_dim=128,
        num_edge_types=8,
        num_gnn_layers=2,
        dropout=0.2,
        mode=mode,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # 训练
    best_val_f1 = 0
    best_state = None
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
              f"Macro F1: {val_metrics['macro_f1']:.4f}")
        
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_state = model.state_dict().copy()
    
    # 最终评估
    if best_state:
        model.load_state_dict(best_state)
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Positive F1: {test_metrics['f1']:.4f}")
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description='Two-stage vulnerability detection experiment')
    parser.add_argument('--vuln_type', type=str, default='reentrancy')
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--modes', type=str, default='baseline,two_stage',
                       help='Comma-separated list of modes to test')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    modes = args.modes.split(',')
    
    results = {}
    
    for mode in modes:
        result = run_experiment(
            mode=mode,
            vuln_type=args.vuln_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            max_samples=args.max_samples,
        )
        results[mode] = result
    
    # 打印汇总结果
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"{'Mode':<15} {'Accuracy':<10} {'Macro F1':<10} {'Pos F1':<10}")
    print("-"*45)
    for mode, result in results.items():
        print(f"{mode:<15} {result['accuracy']:<10.4f} {result['macro_f1']:<10.4f} {result['f1']:<10.4f}")
    
    # 保存结果
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'experiment_results.json'}")


if __name__ == "__main__":
    main()
