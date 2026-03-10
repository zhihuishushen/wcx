"""
使用GNNSCVulDetector原始数据集进行训练
这个数据集是平衡的，可以验证模型在正常数据上的表现
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))


class GNNSCVulDataset(Dataset):
    """GNNSCVulDetector数据集加载器"""
    
    def __init__(self, data_path: str, max_nodes: int = 500):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.max_nodes = max_nodes
        
        # 检查节点特征维度
        sample = self.data[0]['node_features']
        if isinstance(sample, list) and len(sample) > 0:
            if isinstance(sample[0], list):
                self.node_feature_dim = len(sample[0])  # 250
                self.num_graph_nodes = len(sample)  # 可能是3
            else:
                self.node_feature_dim = len(sample)
                self.num_graph_nodes = 1
        else:
            self.node_feature_dim = 250
            self.num_graph_nodes = 1
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        
        # 标签
        label = int(item['targets'])
        
        # 节点特征 - 格式: [[f1, f2, ...], [f1, f2, ...], ...]
        node_features = item['node_features']
        
        # 如果是3x250格式，转置为250x3
        if isinstance(node_features, list) and len(node_features) > 0:
            if isinstance(node_features[0], list):
                # 转置: (num_nodes, feature_dim) -> (feature_dim, num_nodes)
                node_features = np.array(node_features).T  # (feature_dim, num_nodes)
                # 然后再转置回来变成 (num_nodes, feature_dim)
                node_features = node_features.T
            else:
                node_features = np.array([node_features], dtype=np.float32)
        
        if not isinstance(node_features, np.ndarray):
            node_features = np.array(node_features, dtype=np.float32)
            
        # 确保是2D
        if node_features.ndim == 1:
            node_features = node_features.reshape(1, -1)
            
        num_nodes = len(node_features)
        
        # 图结构
        graph = item['graph']  # [[src, dst, type], ...]
        
        # 构建边索引
        edges = []
        edge_types = []
        for e in graph:
            if len(e) >= 3:
                src, dst, etype = e[0], e[1], e[2]
                edges.append([src, dst])
                edge_types.append(etype)
        
        if len(edges) > 0:
            edge_index = np.array(edges, dtype=np.int64).T  # (2, num_edges)
            edge_type = np.array(edge_types, dtype=np.int64)
        else:
            edge_index = np.array([[0], [0]], dtype=np.int64)
            edge_type = np.array([0], dtype=np.int64)
        
        # 填充节点特征到固定维度 (max_nodes, feature_dim)
        feature_dim = node_features.shape[1] if num_nodes > 0 else self.node_feature_dim
        padded_features = np.zeros((self.max_nodes, feature_dim), dtype=np.float32)
        
        actual_nodes = min(num_nodes, self.max_nodes)
        padded_features[:actual_nodes] = node_features[:actual_nodes]
        
        # 填充边
        max_edges = 1000
        padded_edge_index = np.zeros((2, max_edges), dtype=np.int64)
        padded_edge_type = np.zeros(max_edges, dtype=np.int64)
        
        actual_edges = min(len(edges), max_edges)
        padded_edge_index[:, :actual_edges] = edge_index[:, :actual_edges]
        padded_edge_type[:actual_edges] = edge_type[:actual_edges]
        
        return {
            'node_features': torch.from_numpy(padded_features),
            'edge_index': torch.from_numpy(padded_edge_index),
            'edge_type': torch.from_numpy(padded_edge_type),
            'num_nodes': actual_nodes,
            'num_edges': actual_edges,
            'labels': torch.tensor(label, dtype=torch.float32),
            'contract_name': item.get('contract_name', '')
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """自定义batch处理"""
    node_features = torch.stack([b['node_features'] for b in batch])
    edge_index = torch.stack([b['edge_index'] for b in batch])
    edge_type = torch.stack([b['edge_type'] for b in batch])
    num_nodes = torch.tensor([b['num_nodes'] for b in batch])
    num_edges = torch.tensor([b['num_edges'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    
    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_type': edge_type,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'labels': labels
    }


class SimpleGGNN(nn.Module):
    """简化的GGNN层"""
    
    def __init__(self, hidden_dim: int, num_edge_types: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        
        # 消息函数
        self.message_fn = nn.Linear(hidden_dim, hidden_dim)
        
        # 边类型嵌入
        self.edge_type_embed = nn.Embedding(num_edge_types + 1, hidden_dim)
        
        # 更新函数 (GRU)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
    def forward(self, h, edge_index, edge_type, num_nodes):
        """
        Args:
            h: (total_nodes, hidden_dim)
            edge_index: (2, num_edges)
            edge_type: (num_edges,)
            num_nodes: int
        """
        # 初始化消息
        messages = torch.zeros(num_nodes, self.hidden_dim, device=h.device)
        
        if edge_index.size(1) == 0:
            return h
        
        # 获取源节点特征
        src = edge_index[0]
        dst = edge_index[1]
        
        # 边的消息
        src_h = self.message_fn(h[src])
        edge_emb = self.edge_type_embed(edge_type.clamp(0, self.num_edge_types))
        msg = src_h + edge_emb
        
        # 聚合到目标节点
        for i in range(len(dst)):
            messages[dst[i]] += msg[i]
        
        # GRU更新
        h_new = self.gru(messages, h[:num_nodes])
        
        # 创建新tensor而不是inplace修改
        result = torch.zeros_like(h)
        result[:num_nodes] = h_new
        
        return result


class BatchGGNNEncoder(nn.Module):
    """批处理GGNN编码器"""
    
    def __init__(self, node_feature_dim: int = 215, hidden_dim: int = 256, 
                 num_edge_types: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 输入投影
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # GGNN层
        self.gnn_layers = nn.ModuleList([
            SimpleGGNN(hidden_dim, num_edge_types)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_features, edge_index, edge_type, num_nodes):
        """
        Args:
            node_features: (batch, max_nodes, feature_dim)
            edge_index: (batch, 2, max_edges)
            edge_type: (batch, max_edges)
            num_nodes: (batch,)
        """
        batch_size = node_features.size(0)
        max_nodes = node_features.size(1)
        
        # 投影
        h = self.input_proj(node_features)  # (batch, max_nodes, hidden)
        
        # 展平为单个大图
        h_flat = h.view(-1, self.hidden_dim)  # (batch * max_nodes, hidden)
        
        # 添加batch维度到num_nodes
        num_nodes_expanded = num_nodes.unsqueeze(1).expand(-1, max_nodes).contiguous().view(-1)
        
        # 创建每个节点对应的batch索引
        batch_idx = torch.arange(batch_size, device=node_features.device)
        batch_idx = batch_idx.unsqueeze(1).expand(-1, max_nodes).contiguous().view(-1)
        
        # 消息传递
        for layer in self.gnn_layers:
            h_new_list = []
            for b in range(batch_size):
                n = num_nodes[b].item()
                if n == 0:
                    h_new_list.append(torch.zeros(0, self.hidden_dim, device=h_flat.device))
                    continue
                    
                # 该batch的边
                mask = (edge_index[b, 0] < n) & (edge_index[b, 1] < n)
                e_idx = edge_index[b, :, mask]
                e_type = edge_type[b, mask]
                
                if e_idx.size(1) > 0:
                    h_b = layer(h_flat[b*max_nodes:(b+1)*max_nodes].clone(), e_idx, e_type, n)
                else:
                    h_b = h_flat[b*max_nodes:b*max_nodes+n].clone()
                    
                h_new_list.append(h_b)
            
            # 重新组装 - 创建新tensor而不是inplace修改
            h_flat_new = h_flat.clone()
            for b in range(batch_size):
                n = num_nodes[b].item()
                if n > 0:
                    h_flat_new[b*max_nodes:b*max_nodes+n] = h_new_list[b][:n]
            h_flat = h_flat_new
        
        h = self.dropout(h_flat)
        
        # 图级别池化 (sum pooling)
        graph_emb = []
        for b in range(batch_size):
            n = num_nodes[b].item()
            if n > 0:
                emb = h[b*max_nodes:b*max_nodes+n].sum(dim=0)
            else:
                emb = torch.zeros(self.hidden_dim, device=h.device)
            graph_emb.append(emb)
        
        return torch.stack(graph_emb)  # (batch, hidden_dim)


class FocalLoss(nn.Module):
    """Focal Loss"""
    
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha[1], self.alpha[0])
            loss = alpha_t * focal_weight * bce_loss
        else:
            loss = focal_weight * bce_loss
            
        return loss.mean()


class BaselineModel(nn.Module):
    """基线模型 (仅使用下层图)"""
    
    def __init__(self, node_feature_dim: int = 215, hidden_dim: int = 256,
                 num_edge_types: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = BatchGGNNEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_edge_types=num_edge_types,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features, edge_index, edge_type, num_nodes):
        emb = self.encoder(node_features, edge_index, edge_type, num_nodes)
        out = self.classifier(emb)
        return out


def train_epoch(model, dataloader, optimizer, criterion, device, print_every=5):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    n_batches = len(dataloader)
    print(f"\n  [Training] {n_batches} batches", flush=True)
    
    # 使用进度条
    import time
    progress_bar = None
    
    for batch_idx, batch in enumerate(dataloader):
        node_features = batch['node_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_type = batch['edge_type'].to(device)
        num_nodes = batch['num_nodes'].to(device)
        labels = batch['labels'].to(device)
        
        # 打印进度
        pos_count = int(labels.sum().item())
        batch_nodes = int(num_nodes.sum().item())
        
        # 每隔几个batch打印详细状态
        if batch_idx % print_every == 0:
            print(f"    Batch {batch_idx+1}/{n_batches} | nodes:{batch_nodes} | pos:{pos_count}/{len(labels)} | ", end="", flush=True)
        
        start_time = time.time()
        
        optimizer.zero_grad()
        
        outputs = model(node_features, edge_index, edge_type, num_nodes)
        loss = criterion(outputs.squeeze(), labels)
        
        loss.backward()
        optimizer.step()
        
        elapsed = time.time() - start_time
        
        total_loss += loss.item() * len(labels)
        predictions = (outputs.squeeze() > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += len(labels)
        
        if batch_idx % print_every == 0:
            print(f"loss:{loss.item():.4f} | acc:{correct/total:.4f} | time:{elapsed:.2f}s", flush=True)
    
    print(f"  [Train Done] Loss: {total_loss/total:.4f} | Acc: {correct/total:.4f}", flush=True)
    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device, print_every=10):
    model.eval()
    total_loss = 0
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    n_batches = len(dataloader)
    print(f"\n  [Evaluating] {n_batches} batches", flush=True)
    
    import time
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % print_every == 0:
                print(f"    Batch {batch_idx+1}/{n_batches}", end=" | ", flush=True)
                start_time = time.time()
            
            node_features = batch['node_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_type = batch['edge_type'].to(device)
            num_nodes = batch['num_nodes'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(node_features, edge_index, edge_type, num_nodes)
            loss = criterion(outputs.squeeze(), labels)
            
            if batch_idx % print_every == 0:
                elapsed = time.time() - start_time
                print(f"time:{elapsed:.2f}s", flush=True)
            
            total_loss += loss.item() * len(labels)
            
            predictions = (outputs.squeeze() > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.squeeze().cpu().numpy())
    
    print(f"  [Eval Done] Processed {len(all_labels)} samples", flush=True)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = (all_predictions == all_labels).mean()
    
    # 计算F1
    tp = ((all_predictions == 1) & (all_labels == 1)).sum()
    fp = ((all_predictions == 1) & (all_labels == 0)).sum()
    fn = ((all_predictions == 0) & (all_labels == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'loss': total_loss / len(all_labels),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    parser = argparse.ArgumentParser(description='Train with GNNSCVulDetector dataset')
    parser.add_argument('--vuln_type', type=str, default='reentrancy', 
                        choices=['reentrancy', 'timestamp', 'integeroverflow'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_train', type=int, default=None, help='Limit training samples')
    parser.add_argument('--max_val', type=int, default=None, help='Limit validation samples')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Dataset: {args.vuln_type}")
    
    # 加载数据
    data_dir = Path('E:/OneDrive/muti/sc/GNNSCVulDetector/train_data') / args.vuln_type
    
    print("\n[Step 1] Loading dataset...")
    train_dataset = GNNSCVulDataset(str(data_dir / 'train.json'))
    val_dataset = GNNSCVulDataset(str(data_dir / 'valid.json'))
    
    # 限制数据量
    if args.max_train:
        train_dataset.data = train_dataset.data[:args.max_train]
    if args.max_val:
        val_dataset.data = val_dataset.data[:args.max_val]
    
    print(f"  Train: {len(train_dataset)} samples (limited)" if args.max_train else f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples (limited)" if args.max_val else f"  Val: {len(val_dataset)} samples")
    
    # 检查特征维度
    feature_dim = train_dataset.node_feature_dim
    print(f"  Node feature dimension: {feature_dim}")
    
    # 检查标签分布
    train_labels = [train_dataset[i]['labels'].item() for i in range(len(train_dataset))]
    val_labels = [val_dataset[i]['labels'].item() for i in range(len(val_dataset))]
    print(f"  Train positive ratio: {sum(train_labels)/len(train_labels)*100:.1f}%")
    print(f"  Val positive ratio: {sum(val_labels)/len(val_labels)*100:.1f}%")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 创建模型
    print("\n[Step 2] Creating model...")
    model = BaselineModel(
        node_feature_dim=feature_dim,  # 使用数据集的实际特征维度
        hidden_dim=args.hidden_dim,
        num_edge_types=8,
        num_layers=3,
        dropout=args.dropout
    ).to(device)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalLoss(gamma=2.0)
    
    # 训练
    print("\n" + "="*60)
    print("Training Start")
    print("="*60)
    
    best_f1 = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), 'output/gnnsc_model_best.pt')
    
    # 最终评估
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    model.load_state_dict(torch.load('output/gnnsc_model_best.pt'))
    test_metrics = evaluate(model, val_loader, criterion, device)
    
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
