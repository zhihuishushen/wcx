"""
改进的继承相关性预测器
针对数据特点进行优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ImprovedInheritancePredictor(nn.Module):
    """
    改进的继承相关性预测器
    
    改进点：
    1. 更强的继承特征编码
    2. 结合下层图统计特征
    3. 动态阈值调整
    """
    
    def __init__(
        self,
        inheritance_dim: int = 8,
        graph_stat_dim: int = 16,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        # 继承特征编码
        self.inheritance_encoder = nn.Sequential(
            nn.Linear(inheritance_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        # 下层图统计特征编码
        self.graph_stat_encoder = nn.Sequential(
            nn.Linear(graph_stat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        # 融合层
        fusion_dim = hidden_dim  # hidden_dim // 2 + hidden_dim // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # 置信度头
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        inheritance_features: torch.Tensor,
        graph_statistics: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inheritance_features: (batch, 8) 继承特征
            graph_statistics: (batch, 16) 下层图统计特征
            
        Returns:
            inheritance_prob, confidence
        """
        # 编码继承特征
        inh_emb = self.inheritance_encoder(inheritance_features)
        
        # 如果有图统计特征，融合
        if graph_statistics is not None:
            stat_emb = self.graph_stat_encoder(graph_statistics)
            fused = torch.cat([inh_emb, stat_emb], dim=-1)
        else:
            fused = inh_emb
        
        fused = self.fusion(fused)
        
        # 预测
        prob = torch.sigmoid(self.predictor(fused))
        confidence = self.confidence_head(fused)
        
        return prob, confidence


def compute_graph_statistics(node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    计算图的统计特征
    
    Args:
        node_features: (batch, max_nodes, feature_dim)
        edge_index: (2, total_edges)
        
    Returns:
        (batch, 16) 统计特征
    """
    batch_size = node_features.size(0)
    device = node_features.device
    
    stats = []
    
    for i in range(batch_size):
        # 节点数（近似）
        num_nodes = (node_features[i].abs().sum(dim=1) > 0).sum().float()
        
        # 边数
        # 注意：这里需要更精确的边计数
        num_edges = edge_index.size(1) // batch_size
        
        # 节点特征统计
        nf = node_features[i]
        mean_feat = nf.mean(dim=0)
        std_feat = nf.std(dim=0)
        
        stat = torch.cat([
            torch.tensor([num_nodes, num_edges], device=device),
            mean_feat[:7],  # 取前7维
            std_feat[:7],
        ])
        stats.append(stat)
    
    return torch.stack(stats)


class AdaptiveThresholdOptimizer:
    """
    自适应阈值优化器
    
    根据验证集表现动态调整阈值
    """
    
    def __init__(self, initial_threshold: float = 0.5):
        self.threshold = initial_threshold
        self.best_f1 = 0
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        """根据验证集更新阈值"""
        # 尝试不同阈值
        for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
            preds = (predictions > th).float()
            tp = ((preds == 1) & (labels == 1)).sum().item()
            fp = ((preds == 1) & (labels == 0)).sum().item()
            fn = ((preds == 0) & (labels == 1)).sum().item()
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.threshold = th
        
        return self.threshold
