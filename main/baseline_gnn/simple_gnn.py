"""
简化的GGNN编码器
支持批次处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SimpleGGNN(nn.Module):
    """
    简化的GGNN层
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_edge_types: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        
        # 每个边类型一个消息函数
        self.message_funcs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_edge_types)
        ])
        
        # GRU单元用于更新
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: 节点特征 (num_nodes, hidden_dim)
            edge_index: 边索引 (2, num_edges)
            edge_type: 边类型 (num_edges,)
            
        Returns:
            更新后的节点特征
        """
        num_nodes = h.size(0)
        
        # 初始化消息
        messages = torch.zeros(num_nodes, self.hidden_dim, device=h.device)
        
        # 按边类型聚合消息
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        for e_type in range(self.num_edge_types):
            mask = (edge_type == e_type)
            if mask.any():
                src = src_nodes[mask]
                dst = dst_nodes[mask]
                
                # 计算消息
                msg = self.message_funcs[e_type](h[src])
                
                # 累加到目标节点
                messages.index_add_(0, dst, msg)
        
        # 使用GRU更新
        h = self.gru(messages, h)
        
        return h


class BatchGGNNEncoder(nn.Module):
    """
    批次GGNN编码器
    """
    
    def __init__(
        self,
        node_feature_dim: int = 215,
        hidden_dim: int = 256,
        num_edge_types: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 输入投影
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # GNN层
        self.gnn_layers = nn.ModuleList([
            SimpleGGNN(hidden_dim, hidden_dim, num_edge_types)
            for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_nodes_per_graph: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: (batch_size, max_nodes, feature_dim)
            edge_index: (2, total_edges) 全局边索引
            edge_type: (total_edges,)
            num_nodes_per_graph: (batch_size,) 每个图的节点数
            
        Returns:
            graph_embeddings: (batch_size, hidden_dim)
        """
        batch_size = node_features.size(0)
        
        # 展平节点特征
        x = node_features.view(-1, self.node_feature_dim)  # (batch*max_nodes, feature_dim)
        
        # 输入投影
        h = self.input_proj(x)
        
        # 获取有效节点数
        if num_nodes_per_graph is None:
            num_nodes_per_graph = torch.full((batch_size,), h.size(0) // batch_size, device=h.device)
        
        # 相对索引转换（在每个图内）
        # edge_index是全局的，需要转换为相对索引
        max_nodes = node_features.size(1)
        # 创建每个节点的基础偏移
        offsets = torch.arange(batch_size, device=node_features.device) * max_nodes
        offsets = offsets.repeat_interleave(num_nodes_per_graph)
        
        # 注意：这里简化处理，假设edge_index已经是相对索引
        # GNN层
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_type)
            h = self.dropout(h)
        
        # 图池化（求和）
        graph_embeddings = []
        start = 0
        for i, num_nodes in enumerate(num_nodes_per_graph):
            end = start + num_nodes
            graph_emb = h[start:end].sum(dim=0)
            graph_embeddings.append(graph_emb)
            start = end + (max_nodes - num_nodes)  # 跳过padding节点
        
        graph_embeddings = torch.stack(graph_embeddings)
        
        return graph_embeddings


class BaselineModel(nn.Module):
    """
    基线模型：仅使用下层图（代码语义图）
    """
    
    def __init__(
        self,
        node_feature_dim: int = 215,
        hidden_dim: int = 256,
        num_edge_types: int = 8,
        num_gnn_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        
        # 简化的GGNN编码器（支持批次）
        self.encoder = BatchGGNNEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_edge_types=num_edge_types,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, node_features, edge_index, edge_type, num_nodes):
        """
        前向传播
        
        Args:
            node_features: (batch_size, max_nodes, feature_dim)
            edge_index: (2, num_edges)
            edge_type: (num_edges,)
            num_nodes: (batch_size,) 每个图的节点数
            
        Returns:
            logits: (batch_size, 1) 预测概率
        """
        # 编码
        graph_embedding = self.encoder(
            node_features=node_features,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes_per_graph=num_nodes,
        )
        
        # 分类
        logits = self.classifier(graph_embedding)
        return logits
