"""
原版GGNN编码器
复用GNNSCVulDetector的GGNN逻辑，PyTorch实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class GGNNMessagePassing(nn.Module):
    """
    GGNN消息传递层
    
    实现与GNNSCVulDetector相同的消息传递机制：
    - 支持多种边类型
    - GRU聚合更新
    """
    
    def __init__(self, hidden_dim: int, num_edge_types: int, 
                 use_edge_bias: bool = False):
        """
        Args:
            hidden_dim: 隐藏层维度
            num_edge_types: 边类型数量
            use_edge_bias: 是否使用边偏置
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        self.use_edge_bias = use_edge_bias
        
        # 边类型的消息变换权重
        self.edge_weights = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_edge_types)
        ])
        
        # 边偏置（可选）
        if use_edge_bias:
            self.edge_biases = nn.ParameterList([
                nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_edge_types)
            ])
        
        # GRU更新单元
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
    
    def forward(self, 
                node_states: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        """
        消息传递前向传播
        
        Args:
            node_states: 节点状态 [num_nodes, hidden_dim]
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
        
        Returns:
            更新后的节点状态 [num_nodes, hidden_dim]
        """
        num_nodes = node_states.size(0)
        
        # 收集每种类型的消息
        messages_per_type = [[] for _ in range(self.num_edge_types)]
        
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            e_type = edge_type[i].item()
            
            # 计算源节点消息
            msg = self.edge_weights[e_type](node_states[src])
            if self.use_edge_bias:
                msg = msg + self.edge_biases[e_type]
            messages_per_type[e_type].append((dst, msg))
        
        # 聚合每种类型的消息
        aggregated_messages = torch.zeros(num_nodes, self.hidden_dim, device=node_states.device)
        
        for e_type in range(self.num_edge_types):
            if not messages_per_type[e_type]:
                continue
            
            # 按目标节点聚合
            type_messages = defaultdict(list)
            for dst, msg in messages_per_type[e_type]:
                type_messages[dst].append(msg)
            
            for dst, msgs in type_messages.items():
                if msgs:
                    aggregated_messages[dst] = aggregated_messages[dst] + torch.stack(msgs).sum(dim=0)
        
        # GRU更新
        new_states = self.gru(aggregated_messages, node_states)
        
        return new_states


class BasicGNNEncoder(nn.Module):
    """
    基础GNN编码器（简化版GGNN）
    
    适用于没有复杂传播调度的场景
    """
    
    def __init__(self, node_feature_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        """
        Args:
            node_feature_dim: 节点特征维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            dropout: Dropout比率
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 节点特征投影
        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # 边类型嵌入（如果有的话）
        self.edge_type_embed = None
        self.edge_proj = None
        
        # 消息传递层
        self.message_passing = nn.ModuleList([
            GGNNMessagePassing(hidden_dim, num_edge_types=1, use_edge_bias=False)
            for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def set_edge_types(self, num_edge_types: int, edge_feature_dim: int = 0):
        """设置边类型数量"""
        self.message_passing = nn.ModuleList([
            GGNNMessagePassing(self.hidden_dim, num_edge_types, use_edge_bias=False)
            for _ in range(self.num_layers)
        ])
        
        if edge_feature_dim > 0:
            self.edge_type_embed = nn.Embedding(num_edge_types, self.hidden_dim)
            self.edge_proj = nn.Linear(edge_feature_dim + self.hidden_dim, self.hidden_dim)
    
    def forward(self, 
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_features: 节点特征 [num_nodes, node_feature_dim]
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
        
        Returns:
            节点嵌入 [num_nodes, hidden_dim]
        """
        # 特征投影
        h = self.node_proj(node_features)
        h = F.relu(h)
        h = self.dropout(h)
        
        # 多层消息传递
        for layer in self.message_passing:
            if edge_type is not None:
                h = layer(h, edge_index, edge_type)
            else:
                # 如果没有边类型，创建一个全0的边类型
                if edge_index.size(1) > 0:
                    dummy_edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=h.device)
                    h = layer(h, edge_index, dummy_edge_type)
                else:
                    # 没有边时跳过
                    continue
        
        return h


class GGNNEncoder(nn.Module):
    """
    GGNN编码器（完整版）
    
    复用GNNSCVulDetector的完整逻辑：
    - 支持多轮传播
    - 支持传播调度
    - 门控回归输出
    """
    
    def __init__(self, 
                 node_feature_dim: int,
                 hidden_dim: int,
                 num_edge_types: int = 8,
                 propagation_rounds: int = 2,
                 propagation_substeps: int = 20,
                 use_edge_bias: bool = False,
                 dropout: float = 0.1):
        """
        Args:
            node_feature_dim: 节点特征维度
            hidden_dim: 隐藏层维度
            num_edge_types: 边类型数量
            propagation_rounds: 传播轮数
            propagation_substeps: 每轮传播步数
            use_edge_bias: 是否使用边偏置
            dropout: Dropout比率
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        self.propagation_rounds = propagation_rounds
        self.propagation_substeps = propagation_substeps
        
        # 节点特征投影
        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # GGNN消息传递层
        self.gnn_layers = nn.ModuleList([
            GGNNMessagePassing(hidden_dim, num_edge_types, use_edge_bias)
            for _ in range(propagation_rounds)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor,
                initial_nodes: Optional[List[torch.Tensor]] = None,
                sending_nodes: Optional[List[List[torch.Tensor]]] = None,
                receiving_nodes: Optional[List[List[torch.Tensor]]] = None,
                msg_targets: Optional[List[List[torch.Tensor]]] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_features: 节点特征 [num_nodes, node_feature_dim]
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
            initial_nodes: 初始节点索引列表（每轮）
            sending_nodes: 发送节点列表（每轮每步每类型）
            receiving_nodes: 接收节点列表（每轮每步）
            msg_targets: 消息目标节点（每轮每步）
        
        Returns:
            最终节点嵌入 [num_nodes, hidden_dim]
        """
        num_nodes = node_features.size(0)
        device = node_features.device
        
        # 特征投影
        h = self.node_proj(node_features)
        h = F.relu(h)
        h = self.dropout(h)
        
        # 多轮传播
        for prop_round in range(self.propagation_rounds):
            # 使用完整的传播调度或简化版本
            if all(x is not None for x in [initial_nodes, sending_nodes, receiving_nodes, msg_targets]):
                # 使用完整的传播调度
                h = self._propagate_with_schedule(
                    h, prop_round, initial_nodes, sending_nodes, 
                    receiving_nodes, msg_targets, edge_index, edge_type
                )
            else:
                # 简化版本：直接消息传递
                h = self._simplified_propagate(h, edge_index, edge_type)
        
        # 输出投影
        h = self.output_proj(h)
        
        return h
    
    def _simplified_propagate(self,
                             h: torch.Tensor,
                             edge_index: torch.Tensor,
                             edge_type: torch.Tensor) -> torch.Tensor:
        """简化版传播"""
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_type)
            h = F.relu(h)
            h = self.dropout(h)
        return h
    
    def _propagate_with_schedule(self,
                                h: torch.Tensor,
                                prop_round: int,
                                initial_nodes: List[torch.Tensor],
                                sending_nodes: List[List[torch.Tensor]],
                                receiving_nodes: List[List[torch.Tensor]],
                                msg_targets: List[List[torch.Tensor]],
                                full_edge_index: torch.Tensor,
                                full_edge_type: torch.Tensor) -> torch.Tensor:
        """使用GNNSCVulDetector的传播调度"""
        num_nodes = h.size(0)
        device = h.device
        
        # 复制初始节点状态
        new_h = h.clone()
        
        # 初始化初始节点
        if initial_nodes and len(initial_nodes) > prop_round:
            init_nodes = initial_nodes[prop_round]
            new_h[init_nodes] = h[init_nodes]
        
        # 多步传播
        for step in range(self.propagation_substeps):
            # 检查是否有足够的调度信息
            if not all(x is not None and len(x) > step for x in [receiving_nodes]):
                break
            
            recv_nodes = receiving_nodes[prop_round][step]
            if recv_nodes.numel() == 0:
                continue
            
            # 收集消息
            messages = []
            msg_targets_list = []
            
            # 简化：使用所有边
            for i in range(full_edge_index.size(1)):
                src = full_edge_index[0, i]
                dst = full_edge_index[1, i]
                
                if dst in recv_nodes:
                    msg = self.gnn_layers[prop_round].edge_weights[full_edge_type[i]](h[src])
                    messages.append(msg)
                    msg_targets_list.append(dst)
            
            if not messages:
                continue
            
            # 聚合消息
            messages = torch.stack(messages)
            msg_targets_tensor = torch.tensor(msg_targets_list, device=device)
            
            # 按目标节点聚合
            aggregated = torch.zeros(num_nodes, h.size(1), device=device)
            aggregated.index_add_(0, msg_targets_tensor, messages)
            
            # GRU更新
            old_states = h[recv_nodes]
            new_states = self.gnn_layers[prop_round].gru(aggregated[recv_nodes], old_states)
            
            new_h[recv_nodes] = new_states
        
        return new_h


class GatedRegression(nn.Module):
    """
    门控回归层

    与GNNSCVulDetector的gated_regression相同逻辑
    """
    
    def __init__(self, hidden_dim: int, node_feature_dim: int = None):
        """
        Args:
            hidden_dim: 隐藏层维度
            node_feature_dim: 节点特征维度（可选，用于适配不同维度）
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.node_feature_dim = node_feature_dim or hidden_dim

        # 初始特征投影（如果维度不匹配）
        if hidden_dim != self.node_feature_dim:
            self.feature_proj = nn.Linear(self.node_feature_dim, hidden_dim)
        else:
            self.feature_proj = nn.Identity()

        # 门控层：输入 = hidden_dim + hidden_dim = hidden_dim * 2
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # 变换层
        self.transform = nn.Linear(hidden_dim, hidden_dim)

        # 输出层
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self,
                node_embeddings: torch.Tensor,
                initial_features: torch.Tensor,
                graph_nodes_list: Optional[torch.Tensor] = None,
                num_graphs: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            node_embeddings: 节点嵌入 [num_nodes, hidden_dim]
            initial_features: 初始节点特征 [num_nodes, node_feature_dim]
            graph_nodes_list: 每节点所属图的ID [num_nodes]
            num_graphs: 图数量

        Returns:
            graph_repr: 图表示 [num_graphs, hidden_dim] 或标量
            gated_outputs: 门控输出 [num_nodes, hidden_dim]
        """
        # 投影初始特征以匹配 hidden_dim
        projected_features = self.feature_proj(initial_features)

        # 门控计算：concat(node_embeddings, projected_features)
        gate_input = torch.cat([node_embeddings, projected_features], dim=-1)
        gated_outputs = self.gate(gate_input) * torch.tanh(self.transform(node_embeddings))

        if graph_nodes_list is not None and num_graphs is not None:
            # 图级别池化
            graph_repr = torch.zeros(num_graphs, self.hidden_dim, device=node_embeddings.device)
            graph_repr.scatter_add_(0, graph_nodes_list.unsqueeze(-1).expand(-1, self.hidden_dim), gated_outputs)
        else:
            # 整体池化
            graph_repr = gated_outputs.mean(dim=0, keepdim=True)

        # 最终预测
        prediction = self.output(graph_repr)

        return prediction.squeeze(-1), graph_repr


class BaselineGGNNEncoder(nn.Module):
    """
    基线GGNN编码器（完整版）
    
    整合：
    1. GGNN编码器
    2. 门控回归
    3. 图级别池化
    """
    
    def __init__(self, 
                 node_feature_dim: int,
                 hidden_dim: int = 128,
                 num_edge_types: int = 8,
                 propagation_rounds: int = 2,
                 propagation_substeps: int = 20,
                 dropout: float = 0.1):
        """
        Args:
            node_feature_dim: 节点特征维度
            hidden_dim: 隐藏层维度
            num_edge_types: 边类型数量
            propagation_rounds: 传播轮数
            propagation_substeps: 每轮传播步数
            dropout: Dropout比率
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        
        # GGNN编码器
        self.gnn = GGNNEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_edge_types=num_edge_types,
            propagation_rounds=propagation_rounds,
            propagation_substeps=propagation_substeps,
            dropout=dropout
        )
        
        # 门控回归
        self.gated_reg = GatedRegression(hidden_dim, node_feature_dim)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor,
                graph_nodes_list: Optional[torch.Tensor] = None,
                num_graphs: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            node_features: 节点特征 [num_nodes, node_feature_dim]
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
            graph_nodes_list: 每节点所属图的ID [num_nodes]
            num_graphs: 图数量
        
        Returns:
            prediction: 预测值 [num_graphs] 或 [1]
            graph_embedding: 图嵌入 [num_graphs, hidden_dim]
        """
        # GGNN编码
        node_embeddings = self.gnn(
            node_features, edge_index, edge_type
        )
        
        # 门控回归
        prediction, graph_embedding = self.gated_reg(
            node_embeddings, node_features, graph_nodes_list, num_graphs
        )
        
        # 分类
        logits = self.classifier(graph_embedding)
        
        return logits.squeeze(-1), graph_embedding
    
    def predict(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor,
                graph_nodes_list: Optional[torch.Tensor] = None,
                num_graphs: Optional[int] = None) -> torch.Tensor:
        """
        预测（带sigmoid）
        """
        logits, _ = self.forward(node_features, edge_index, edge_type, graph_nodes_list, num_graphs)
        return torch.sigmoid(logits)


class SimpleGraphEncoder(nn.Module):
    """
    简化图编码器（用于测试）
    
    使用PyG风格的消息传递
    """
    
    def __init__(self, node_feature_dim: int, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.1):
        """
        Args:
            node_feature_dim: 节点特征维度
            hidden_dim: 隐藏层维度
            num_layers: 消息传递层数
            dropout: Dropout比率
        """
        super().__init__()
        
        # 节点特征投影
        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # 消息传递层
        self.convs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # 注意力权重
        self.attentions = nn.ModuleList([
            nn.Linear(hidden_dim * 2, 1) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
    
    def message_pass(self, 
                     h: torch.Tensor,
                     edge_index: torch.Tensor) -> torch.Tensor:
        """单层消息传递"""
        num_nodes = h.size(0)
        
        # 计算注意力权重
        src, dst = edge_index
        edge_h = torch.cat([h[src], h[dst]], dim=-1)
        attn = F.leaky_relu(self.attentions[0](edge_h), 0.2)
        attn = F.softmax(attn, dim=0)
        
        # 消息聚合
        messages = torch.zeros(num_nodes, h.size(1), device=h.device)
        messages.scatter_add_(0, dst.unsqueeze(-1).expand(-1, h.size(1)), attn * h[src])
        
        return messages
    
    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_features: [num_nodes, node_feature_dim]
            edge_index: [2, num_edges]
            edge_type: [num_edges] (可选)
        
        Returns:
            节点嵌入 [num_nodes, hidden_dim]
        """
        h = self.node_proj(node_features)
        
        for i in range(self.num_layers):
            # 消息传递
            src, dst = edge_index
            edge_h = torch.cat([h[src], h[dst]], dim=-1)
            attn = F.leaky_relu(self.attentions[i](edge_h), 0.2)
            attn = F.softmax(attn, dim=0)
            
            messages = torch.zeros(h.size(0), h.size(1), device=h.device)
            messages.scatter_add_(0, dst.unsqueeze(-1).expand(-1, h.size(1)), attn * h[src])
            
            # 更新
            h = h + F.relu(self.convs[i](messages))
            h = self.dropout(h)
        
        return h


