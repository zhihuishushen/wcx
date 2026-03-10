"""
双层GNN模型训练脚本
下层图：代码语义图（GGNN）
上层图：继承关系图（通过UpperGraphBuilder构建）
自适应门控：决定是否使用上层图信息
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
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from main.gnn_core.dataset import DatasetLoader, ContractGraphDataset, create_data_loaders, collate_fn
from main.upper_graph.builder import UpperGraphBuilder
from main.baseline_gnn.simple_gnn import SimpleGGNN, BatchGGNNEncoder
from balance_dataset import balance_dataset, load_vulnerability_data


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
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


class UpperGraphEncoder(nn.Module):
    """上层图编码器（继承关系图）"""
    
    def __init__(
        self,
        feature_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        
        # 简单的GCN层
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
    def forward(self, node_features, edge_index, num_nodes):
        """
        Args:
            node_features: (num_upper_nodes, feature_dim)
            edge_index: (2, num_edges)
            num_nodes: int
        """
        # 输入投影
        h = F.relu(self.input_proj(node_features))
        
        # 简单的消息传递
        for layer in self.layers:
            h_new = torch.zeros_like(h)
            if edge_index.size(1) > 0:
                src, dst = edge_index
                # 聚合邻居信息
                neighbor_msg = torch.zeros_like(h)
                for i in range(len(dst)):
                    neighbor_msg[dst[i]] += h[src[i]]
                h_new = h + F.relu(layer(neighbor_msg))
            h = h_new
        
        # 如果没有节点，返回零向量
        if num_nodes == 0:
            return torch.zeros(self.hidden_dim, device=node_features.device)
        
        return h.mean(dim=0)  # (hidden_dim,)


class AdaptiveGate(nn.Module):
    """自适应门控"""
    
    def __init__(self, graph_dim: int, upper_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # 只基于下层图特征决定是否使用上层图
        self.gate_net = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, graph_emb):
        """
        Args:
            graph_emb: (batch, graph_dim)
        Returns:
            use_upper: (batch, 1) - 使用上层图的概率
        """
        return self.gate_net(graph_emb)


class DualLayerGNNModel(nn.Module):
    """双层GNN模型"""
    
    def __init__(
        self,
        node_feature_dim: int = 215,
        upper_feature_dim: int = 6,
        hidden_dim: int = 256,
        upper_hidden_dim: int = 64,
        num_edge_types: int = 8,
        num_gnn_layers: int = 3,
        dropout: float = 0.1,
        use_upper_graph: bool = True,
    ):
        super().__init__()
        
        self.use_upper_graph = use_upper_graph
        
        # 下层图编码器
        self.lower_encoder = BatchGGNNEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_edge_types=num_edge_types,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )
        
        # 上层图编码器（如果启用）
        if use_upper_graph:
            self.upper_encoder = UpperGraphEncoder(
                feature_dim=upper_feature_dim,
                hidden_dim=upper_hidden_dim,
            num_layers=2,
        )
        
        # 自适应门控
        self.gate = AdaptiveGate(hidden_dim, upper_hidden_dim)
        
        # 分类器
        # 输入：下层图 + 融合后的特征
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        lower_node_features,
        lower_edge_index,
        lower_edge_type,
        lower_num_nodes,
        upper_node_features=None,
        upper_edge_index=None,
        upper_num_nodes=0,
    ):
        """
        Args:
            lower_*: 下层图（代码语义图）
            upper_*: 上层图（继承关系图）
        """
        # 下层图编码
        lower_emb = self.lower_encoder(
            lower_node_features,
            lower_edge_index,
            lower_edge_type,
            lower_num_nodes
        )  # (batch, hidden_dim)
        
        # 判断是否使用上层图（如果禁用，则强制为0）
        if self.use_upper_graph:
            use_upper_prob = self.gate(lower_emb)  # (batch, 1)
        else:
            use_upper_prob = torch.zeros(lower_emb.size(0), 1, device=lower_emb.device)
        
        # 上层图编码（如果有且启用）
        if self.use_upper_graph and upper_node_features is not None and upper_num_nodes > 0:
            upper_emb = self.upper_encoder(
                upper_node_features,
                upper_edge_index,
                upper_num_nodes
            )  # (hidden_dim,)
            upper_emb = upper_emb.unsqueeze(0).expand(lower_emb.size(0), -1)  # (batch, upper_hidden_dim)
            
            # 融合
            # 下层图始终参与，上层图根据门控概率参与
            upper_emb_expanded = torch.zeros_like(lower_emb)
            upper_emb_expanded[:, :upper_emb.size(1)] = upper_emb
            
            # 融合策略：(1 - use_upper) * lower + use_upper * (lower + upper)
            fused = lower_emb + use_upper_prob * upper_emb_expanded
        else:
            # 没有上层图或禁用，只用下层图
            fused = lower_emb
        
        # 分类
        # 拼接下层图和融合特征
        combined = torch.cat([lower_emb, fused], dim=-1)
        output = self.classifier(combined)
        
        return output, use_upper_prob


def build_upper_graph_data(contract_name, all_contracts, loader):
    """构建上层图数据"""
    # 提取所有合约的继承关系
    contracts_for_graph = []
    for c in all_contracts:
        if c.parent_contracts:
            contracts_for_graph.append({
                'name': c.contract_name,
                'parents': c.parent_contracts
            })
    
    if not contracts_for_graph:
        return None
    
    builder = UpperGraphBuilder(max_nodes=50)
    upper_data = builder.build_upper_graph_sample(contract_name, contracts_for_graph)
    
    return upper_data


def train_epoch(model, dataloader, optimizer, criterion, device, all_contracts, loader, print_gate_stats=True):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    n_batches = len(dataloader)
    print(f"\n  [Training] {n_batches} batches", flush=True)
    
    # 统计门控输出
    gate_stats = {'use_upper_sum': 0, 'use_upper_count': 0, 'use_upper_values': []}
    
    for batch_idx, batch in enumerate(dataloader):
        node_features = batch['node_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_type = batch['edge_type'].to(device)
        labels = batch['labels'].to(device)
        num_nodes = batch['num_nodes']
        
        # 打印进度
        pos_count = labels.sum().item()
        if batch_idx % 10 == 0:
            print(f"    Batch {batch_idx+1}/{n_batches} | nodes:{num_nodes.sum().item()} | pos:{pos_count}/{len(labels)}", flush=True)
        
        # 创建batch索引
        batch_node_idx = torch.repeat_interleave(
            torch.arange(len(num_nodes), device=device),
            num_nodes.to(device)
        )
        
        optimizer.zero_grad()
        
        # 前向传播（暂时只使用下层图）
        outputs, use_upper = model(
            node_features, edge_index, edge_type, num_nodes
        )
        
        # 统计门控输出
        if print_gate_stats:
            gate_stats['use_upper_sum'] += use_upper.sum().item()
            gate_stats['use_upper_count'] += len(use_upper)
            use_upper_list = use_upper.squeeze().tolist()
            if isinstance(use_upper_list, float):
                use_upper_list = [use_upper_list]
            gate_stats['use_upper_values'].extend(use_upper_list)
        
        # 计算损失
        outputs_flat = outputs.view(-1)
        labels_flat = labels.float().view(-1)
        loss = criterion(outputs_flat, labels_flat)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels_flat)
        
        predictions = (outputs_flat > 0.5).float()
        correct += (predictions == labels_flat).sum().item()
        total += len(labels_flat)
        
        if batch_idx % 20 == 0:
            print(f"    Loss: {loss.item():.4f} | Acc: {correct/total:.4f}", flush=True)
    
    # 打印门控统计
    if print_gate_stats and gate_stats['use_upper_count'] > 0:
        avg_use_upper = gate_stats['use_upper_sum'] / gate_stats['use_upper_count']
        print(f"  [Gate Stats] Avg use_upper: {avg_use_upper:.4f} | "
              f"Min: {min(gate_stats['use_upper_values']):.4f} | "
              f"Max: {max(gate_stats['use_upper_values']):.4f}", flush=True)
    
    print(f"  [Train Done] Loss: {total_loss/total:.4f} | Acc: {correct/total:.4f}", flush=True)
    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
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
            
            outputs, _ = model(node_features, edge_index, edge_type, num_nodes)
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
    
    accuracy = (all_predictions == all_labels).mean()
    
    def compute_class_metrics(predictions, labels, positive_class=1):
        tp = ((predictions == positive_class) & (labels == positive_class)).sum()
        fp = ((predictions == positive_class) & (labels != positive_class)).sum()
        fn = ((predictions != positive_class) & (labels == positive_class)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    neg_p, neg_r, neg_f1 = compute_class_metrics(all_predictions, all_labels, 0)
    pos_p, pos_r, pos_f1 = compute_class_metrics(all_predictions, all_labels, 1)
    macro_f1 = (neg_f1 + pos_f1) / 2
    
    # 打印预测分布
    num_pred_pos = (all_predictions == 1).sum()
    num_pred_neg = (all_predictions == 0).sum()
    num_actual_pos = (all_labels == 1).sum()
    num_actual_neg = (all_labels == 0).sum()
    print(f"  [Prediction Dist] Predicted pos: {num_pred_pos}, neg: {num_pred_neg} | "
          f"Actual pos: {num_actual_pos}, neg: {num_actual_neg}")
    
    return {
        'loss': total_loss / len(all_labels),
        'accuracy': accuracy,
        'precision_pos': pos_p,
        'recall_pos': pos_r,
        'f1_pos': pos_f1,
        'macro_f1': macro_f1,
    }


def main():
    parser = argparse.ArgumentParser(description='Train Dual-Layer GNN Model')
    parser.add_argument('--vuln_type', type=str, default='reentrancy')
    parser.add_argument('--max_samples', type=int, default=None, help='None means use all data')
    parser.add_argument('--balance_ratio', type=float, default=None, 
                        help='Balance ratio for undersampling. None=original, 1.0=1:1, 2.0=1:2')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_upper_graph', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether to use upper graph (inheritance): true or false')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # ============================================================
    # [Step 1] 加载和处理数据（支持平衡处理）
    # ============================================================
    print("\n[Step 1] Loading dataset...", flush=True)
    
    # 加载原始数据信息（用于平衡处理）
    raw_df = load_vulnerability_data(args.vuln_type)
    print(f"  Raw data: {len(raw_df)} contracts (positive={raw_df['ground truth'].sum()}, negative={(1-raw_df['ground truth']).sum()})")
    
    # 如果指定了平衡比例，则进行平衡处理
    if args.balance_ratio is not None:
        print(f"  [BALANCE] Applying undersampling with ratio={args.balance_ratio}...", flush=True)
        balanced_df = balance_dataset(args.vuln_type, ratio=args.balance_ratio, random_state=args.seed)
        print(f"  Balanced: {len(balanced_df)} contracts", flush=True)
        
        # 使用平衡后的数据创建数据加载器
        # 获取平衡后的文件列表
        balanced_files = balanced_df['file'].tolist()
        balanced_contracts_list = balanced_df['contract'].tolist()
        balanced_labels = balanced_df['ground truth'].tolist()
        
        # 加载合约数据
        loader = DatasetLoader()
        all_contracts = loader.load_dataset(args.vuln_type)
        
        # 筛选出平衡数据中包含的合约
        filtered_contracts = []
        for c in all_contracts:
            if c.file_id in balanced_files:
                # 找到对应的contract和label
                idx = None
                for i, (f, con) in enumerate(zip(balanced_df['file'], balanced_df['contract'])):
                    if f == c.file_id and con == c.contract_name:
                        idx = i
                        break
                if idx is not None:
                    c.label = balanced_labels[idx]
                    filtered_contracts.append(c)
        
        contracts = filtered_contracts
        print(f"  Filtered contracts: {len(contracts)}")
        
    else:
        # 使用原始数据
        train_loader, val_loader, test_loader = create_data_loaders(
            vuln_type=args.vuln_type,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        contracts = None
    
    # 如果不使用平衡处理，使用原始的 create_data_loaders
    if args.balance_ratio is None:
        train_loader, val_loader, test_loader = create_data_loaders(
            vuln_type=args.vuln_type,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        loader = DatasetLoader()
        all_contracts = loader.load_dataset(args.vuln_type)
    else:
        # 手动划分平衡后的数据
        np.random.seed(args.seed)
        indices = np.random.permutation(len(contracts))
        
        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
        train_size = int(len(indices) * train_ratio)
        val_size = int(len(indices) * val_ratio)
        
        train_contracts = [contracts[i] for i in indices[:train_size]]
        val_contracts = [contracts[i] for i in indices[train_size:train_size + val_size]]
        test_contracts = [contracts[i] for i in indices[train_size + val_size:]]
        
        max_nodes = 500
        train_dataset = ContractGraphDataset(train_contracts, args.vuln_type, max_nodes=max_nodes)
        val_dataset = ContractGraphDataset(val_contracts, args.vuln_type, max_nodes=max_nodes)
        test_dataset = ContractGraphDataset(test_contracts, args.vuln_type, max_nodes=max_nodes)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"  Loaded: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}", flush=True)
    print("[Step 1] Done!\n", flush=True)
    
    # 获取所有合约用于构建上层图（如果在Step1中没有加载）
    print("[Step 2] Loading contracts for upper graph...", flush=True)
    if contracts is None:
        loader = DatasetLoader()
        all_contracts = loader.load_dataset(args.vuln_type)
    print(f"  Total contracts: {len(all_contracts)}", flush=True)
    
    # 统计有继承关系的合约
    has_inheritance = sum(1 for c in all_contracts if c.inheritance_depth > 0)
    print(f"  With inheritance: {has_inheritance}/{len(all_contracts)}", flush=True)
    print("[Step 2] Done!\n", flush=True)
    
    # 计算类别权重 - 使用更激进的权重来处理严重不平衡
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['labels'].numpy())
    all_labels = np.array(all_labels)
    
    num_pos = sum(all_labels)
    num_neg = len(all_labels) - num_pos
    
    # 更激进的权重计算
    if num_pos > 0:
        # 正样本权重 = 负样本数/正样本数 * 平衡因子
        weight_pos = (num_neg / num_pos) * 2.0  # 乘以2.0作为平衡因子
    else:
        weight_pos = 10.0  # 如果没有正样本，使用默认高权重
    
    weight_neg = 1.0  # 负样本权重保持为1
    
    class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32).to(device)
    print(f"Class weights: neg={weight_neg:.2f}, pos={weight_pos:.2f}")
    print(f"Data balance: {num_pos} positive / {num_neg} negative = {num_pos/len(all_labels)*100:.2f}%")
    
    # 创建模型
    print("[Step 3] Creating model...", flush=True)
    model = DualLayerGNNModel(
        node_feature_dim=215,
        hidden_dim=args.hidden_dim,
        num_edge_types=8,
        dropout=args.dropout,
        use_upper_graph=args.use_upper_graph,
    ).to(device)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    if args.use_upper_graph:
        print("  Mode: Dual-Layer GNN (with upper graph)", flush=True)
    else:
        print("  Mode: Baseline GNN (without upper graph)", flush=True)
    print("[Step 3] Done!\n", flush=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 使用更强的Focal Loss参数来处理严重不平衡
    criterion = FocalLoss(alpha=class_weights, gamma=3.0)
    
    best_val_f1 = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_macro_f1': []}
    
    print("="*60)
    print("Training Start")
    print("="*60)
    
    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]", flush=True)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, all_contracts, loader
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
        
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            torch.save(model.state_dict(), 'output/dual_layer_model_best.pt')
    
    # 最终评估
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    model.load_state_dict(torch.load('output/dual_layer_model_best.pt'))
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Positive Class - Precision: {test_metrics['precision_pos']:.4f}, "
          f"Recall: {test_metrics['recall_pos']:.4f}, F1: {test_metrics['f1_pos']:.4f}")
    
    with open('output/dual_layer_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
