"""
基线模型训练脚本
使用GGNN编码器（仅下层图）进行漏洞检测
继承自GNNSCVulDetector的超参数
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
from typing import Dict, Tuple
from tqdm import tqdm

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

from main.gnn_core.dataset import DatasetLoader, ContractGraphDataset, create_data_loaders
from main.baseline_gnn.simple_gnn import BaselineModel


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    计算类别权重，处理不平衡数据
    
    Args:
        labels: 标签数组
        
    Returns:
        类别权重
    """
    num_positive = np.sum(labels == 1)
    num_negative = np.sum(labels == 0)
    total = len(labels)
    
    weight_positive = total / (2 * num_positive) if num_positive > 0 else 1.0
    weight_negative = total / (2 * num_negative) if num_negative > 0 else 1.0
    
    return torch.tensor([weight_negative, weight_positive], dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Focal Loss - 用于处理类别不平衡
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 预测概率 (batch_size, 1)
            targets: 真实标签 (batch_size, 1)
        """
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Focal Loss: (1 - p)^gamma * log(p)
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_weight = (1 - pt) ** self.gamma
        
        loss = focal_weight * bce_loss
        
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha[1], self.alpha[0])
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = 'cuda',
) -> Tuple[float, float]:
    """
    训练一个epoch
    
    Returns:
        (loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    n_batches = len(dataloader)
    print(f"\n  [Training] {n_batches} batches", flush=True)
    
    for batch_idx, batch in enumerate(dataloader):
        # 获取数据
        node_features = batch['node_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_type = batch['edge_type'].to(device)
        labels = batch['labels'].to(device)
        num_nodes = batch['num_nodes']
        
        # 打印进度
        if batch_idx % 10 == 0:
            pos_count = int(labels.sum().item())
            print(f"    Batch {batch_idx+1}/{n_batches} | nodes:{int(num_nodes.sum().item())} | pos:{pos_count}/{len(labels)}", flush=True)
        
        # 创建batch索引
        batch_node_idx = torch.repeat_interleave(
            torch.arange(len(num_nodes), device=device),
            num_nodes.to(device)
        )
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(node_features, edge_index, edge_type, num_nodes)
        
        # 计算损失
        loss = criterion(outputs.squeeze(), labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        
        # 计算准确率
        predictions = (outputs.squeeze() > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += len(labels)
        
        if batch_idx % 20 == 0:
            print(f"    Loss: {loss.item():.4f} | Acc: {correct/total:.4f}", flush=True)
    
    print(f"  [Train Done] Loss: {total_loss/total:.4f} | Acc: {correct/total:.4f}", flush=True)
    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = 'cuda',
) -> Dict:
    """
    评估模型
    
    Returns:
        评估指标字典
    """
    model.eval()
    total_loss = 0
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    n_batches = len(dataloader)
    print(f"\n  [Evaluating] {n_batches} batches", flush=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 20 == 0:
                print(f"    Batch {batch_idx+1}/{n_batches}", flush=True)
            node_features = batch['node_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_type = batch['edge_type'].to(device)
            labels = batch['labels'].to(device)
            num_nodes = batch['num_nodes']
            
            outputs = model(node_features, edge_index, edge_type, num_nodes)
            loss = criterion(outputs.squeeze(), labels)
            
            total_loss += loss.item() * len(labels)
            
            predictions = (outputs.squeeze() > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.squeeze().cpu().numpy())
    
    print(f"  [Eval Done] Processed {len(all_labels)} samples", flush=True)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算指标
    accuracy = (all_predictions == all_labels).mean()
    
    # 计算每个类的Precision, Recall, F1
    def compute_class_metrics(predictions, labels, positive_class=1):
        tp = ((predictions == positive_class) & (labels == positive_class)).sum()
        fp = ((predictions == positive_class) & (labels != positive_class)).sum()
        fn = ((predictions != positive_class) & (labels == positive_class)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    # 负类指标
    neg_p, neg_r, neg_f1 = compute_class_metrics(all_predictions, all_labels, 0)
    # 正类指标
    pos_p, pos_r, pos_f1 = compute_class_metrics(all_predictions, all_labels, 1)
    
    # Macro F1
    macro_f1 = (neg_f1 + pos_f1) / 2
    
    return {
        'loss': total_loss / len(all_labels),
        'accuracy': accuracy,
        'precision_neg': neg_p,
        'recall_neg': neg_r,
        'f1_neg': neg_f1,
        'precision_pos': pos_p,
        'recall_pos': pos_r,
        'f1_pos': pos_f1,
        'macro_f1': macro_f1,
    }


def main():
    parser = argparse.ArgumentParser(description='Train baseline model')
    parser.add_argument('--vuln_type', type=str, default='reentrancy',
                        help='Vulnerability type')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples (for testing)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate (from GNNSCVulDetector)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension (from GNNSCVulDetector)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 加载数据
    print(f"\nLoading {args.vuln_type} dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(
        vuln_type=args.vuln_type,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    
    # 计算类别权重
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['labels'].numpy())
    all_labels = np.array(all_labels)
    class_weights = compute_class_weights(all_labels).to(device)
    print(f"Class weights: neg={class_weights[0]:.2f}, pos={class_weights[1]:.2f}")
    
    # 创建模型
    model = BaselineModel(
        node_feature_dim=215,  # 与GNNSCVulDetector一致
        hidden_dim=args.hidden_dim,  # 256 (from GNNSCVulDetector)
        num_edge_types=8,
        num_gnn_layers=3,
        dropout=args.dropout,
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器 (来自GNNSCVulDetector)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 损失函数 (Focal Loss处理不平衡)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # 训练
    best_val_f1 = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_macro_f1': [],
    }
    
    print("\n" + "="*60)
    print("Training Baseline Model (Lower Graph Only)")
    print("="*60)
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
              f"Macro F1: {val_metrics['macro_f1']:.4f}")
        
        # 保存最佳模型
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            torch.save(model.state_dict(), 'output/baseline_model_best.pt')
    
    # 最终评估
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    model.load_state_dict(torch.load('output/baseline_model_best.pt'))
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Positive Class - Precision: {test_metrics['precision_pos']:.4f}, "
          f"Recall: {test_metrics['recall_pos']:.4f}, F1: {test_metrics['f1_pos']:.4f}")
    
    # 保存训练历史
    with open('output/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
