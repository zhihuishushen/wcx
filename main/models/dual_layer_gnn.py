"""
双层图漏洞检测模型

整合：
1. 基线GGNN编码器（下层图 - 代码语义）
2. 上层图编码器（继承关系）
3. 可学习门控（决定何时使用上层图）
4. 预测融合（结合两种表示）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from ..baseline_gnn import BaselineGGNNEncoder, GGNNEncoder
from ..gating import AdaptiveGate, ConfidenceCalculator
from ..upper_graph.fusion import PredictionFusion, HierarchicalFusion


class DualLayerGNNModel(nn.Module):
    """
    双层图漏洞检测模型
    
    核心思想：
    - 下层图：代码内部语义（复现GNNSCVulDetector）- 始终参与预测
    - 上层图：继承关系图（可选）- 通过门控决定是否使用
    - 可学习门控：仅基于下层图embedding预测是否需要上层图
    
    关键设计：
    1. 下层图始终参与预测（保底精度）
    2. 门控只使用下层图特征做决策，不依赖上层图信息
    3. 门控决定的是上层图特征的权重，而非完全替代
    """
    
    def __init__(self,
                 node_feature_dim: int,
                 hidden_dim: int = 128,
                 num_classes: int = 4,
                 num_edge_types: int = 8,
                 upper_graph_dim: int = 64,
                 use_upper_graph: bool = True,
                 gate_type: str = 'adaptive',  # 'adaptive', 'simple', 'always', 'never'
                 **kwargs):
        """
        Args:
            node_feature_dim: 节点特征维度
            hidden_dim: 隐藏层维度
            num_classes: 漏洞类型数量
            num_edge_types: 边类型数量
            upper_graph_dim: 上层图嵌入维度
            use_upper_graph: 是否使用上层图
            gate_type: 门控类型
                - 'adaptive': 学习型门控（默认）
                - 'simple': 简单加权融合
                - 'always': 始终使用上层图
                - 'never': 从不使用上层图
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_upper_graph = use_upper_graph
        self.gate_type = gate_type
        
        # ========== 下层图编码器（基线 - 始终参与） ==========
        self.lower_encoder = BaselineGGNNEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_edge_types=num_edge_types,
            propagation_rounds=2,
            propagation_substeps=20,
            dropout=0.3
        )
        
        # ========== 上层图编码器（继承关系） ==========
        if use_upper_graph:
            self.upper_encoder = UpperGraphEncoder(
                input_dim=upper_graph_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim
            )
            
            # ========== 可学习门控 ==========
            # 门控只接收下层图embedding，预测是否需要上层图
            self.gate = AdaptiveGate(
                graph_dim=hidden_dim,
                inheritance_dim=hidden_dim,  # 用于融合预测
                hidden_dim=hidden_dim,
                num_heads=4
            )
        
        # ========== 预测融合 ==========
        self.prediction_fusion = PredictionFusion(hidden_dim=hidden_dim)
        
        # ========== 多标签分类头 ==========
        # 输入是融合后的特征: lower + gated_upper
        self.classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_classes)
        ])
        
        # ========== 下游任务分类头（风险等级） ==========
        self.risk_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 3)  # low/medium/high
        )
        
        # ========== 下层图专用分类头（用于保底预测） ==========
        self.lower_classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_classes)
        ])
        
    def forward(self,
                lower_graph: Dict[str, torch.Tensor],
                upper_graph: Optional[Dict[str, torch.Tensor]] = None,
                has_upper: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        关键设计：
        1. 下层图始终参与，产生 lower_embedding
        2. 如果 has_upper=True，则有条件地使用上层图
        3. 门控只基于 lower_embedding 做决策
        4. 最终预测 = 门控加权融合(下层图预测, 上层图预测)
        
        Args:
            lower_graph: 下层图数据
                - node_features: 节点特征 [num_nodes, node_feature_dim]
                - edge_index: 边索引 [2, num_edges]
                - edge_type: 边类型 [num_edges]
                - graph_nodes_list: 图ID [num_nodes]
                - num_graphs: 图数量 int
            upper_graph: 上层图数据（可选）
            has_upper: 是否有上层图可用
        
        Returns:
            包含以下键的字典：
            - predictions: 多标签预测 [num_graphs, num_classes]
            - risk_levels: 风险等级 [num_graphs, 3]
            - use_upper: 门控信号 [num_graphs, 1]
            - confidence: 置信度 [num_graphs, 1]
            - lower_embedding: 下层图嵌入 [num_graphs, hidden_dim]
            - upper_embedding: 上层图嵌入 [num_graphs, hidden_dim] (如果使用)
            - fused_embedding: 融合嵌入 [num_graphs, hidden_dim * 2]
            - lower_predictions: 下层图单独的预测（保底）
        """
        results = {}
        
        # ========== 下层图编码（始终执行）==========
        lower_pred, lower_embedding = self.lower_encoder(
            node_features=lower_graph['node_features'],
            edge_index=lower_graph['edge_index'],
            edge_type=lower_graph['edge_type'],
            graph_nodes_list=lower_graph.get('graph_nodes_list'),
            num_graphs=lower_graph.get('num_graphs', 1)
        )
        results['lower_embedding'] = lower_embedding
        results['lower_pred'] = lower_pred
        
        # 下层图单独的预测（保底）
        lower_predictions = []
        for i, classifier in enumerate(self.lower_classifier):
            pred = classifier(lower_embedding)
            lower_predictions.append(pred)
        results['lower_predictions'] = torch.cat(lower_predictions, dim=-1)
        
        # ========== 上层图处理（可选）==========
        use_upper_value = 0.0  # 默认不使用上层图
        
        if self.use_upper_graph and has_upper and upper_graph is not None:
            # 上层图编码
            upper_embedding = self.upper_encoder(upper_graph)
            results['upper_embedding'] = upper_embedding
            
            # 门控决策 - 仅基于下层图embedding
            if self.gate_type == 'adaptive':
            use_upper, confidence, gate_info = self.gate(lower_embedding, upper_embedding)
            results['use_upper'] = use_upper
            results['confidence'] = confidence
            results['gate_info'] = gate_info
                use_upper_value = use_upper.mean().item() if use_upper.dim() > 1 else use_upper.item()
                
            elif self.gate_type == 'always':
                use_upper = torch.ones_like(lower_embedding[:, :1])
                results['use_upper'] = use_upper
                results['confidence'] = torch.ones_like(lower_embedding[:, :1])
                use_upper_value = 1.0
                
            elif self.gate_type == 'never':
                use_upper = torch.zeros_like(lower_embedding[:, :1])
                results['use_upper'] = use_upper
                results['confidence'] = torch.zeros_like(lower_embedding[:, :1])
                use_upper_value = 0.0
                
            else:  # simple
                use_upper = torch.full((lower_embedding.size(0), 1), 0.5, device=lower_embedding.device)
                results['use_upper'] = use_upper
                results['confidence'] = torch.full((lower_embedding.size(0), 1), 0.5, device=lower_embedding.device)
                use_upper_value = 0.5
            
            # 门控加权融合
            # fused = lower + use_upper * upper
            upper_weighted = use_upper * upper_embedding if use_upper.dim() > 1 else use_upper.view(-1, 1) * upper_embedding
            fused_embedding = torch.cat([lower_embedding, lower_embedding + upper_weighted], dim=-1)
            
        else:
            # 不使用上层图 - 仅使用下层图
            results['use_upper'] = torch.zeros_like(lower_embedding[:, :1])
            results['confidence'] = torch.zeros_like(lower_embedding[:, :1])
            results['upper_embedding'] = torch.zeros_like(lower_embedding)
            # fused = [lower, lower]
            fused_embedding = torch.cat([lower_embedding, lower_embedding], dim=-1)
        
        results['fused_embedding'] = fused_embedding
        
        # ========== 多标签分类（基于融合特征）==========
        predictions = []
        for i, classifier in enumerate(self.classifier):
            pred = classifier(fused_embedding)
            predictions.append(pred)
        results['predictions'] = torch.cat(predictions, dim=-1)  # [num_graphs, num_classes]
        
        # ========== 风险等级分类 ==========
        risk_levels = self.risk_classifier(fused_embedding)
        results['risk_levels'] = risk_levels
        
        return results
    
    def predict(self,
                lower_graph: Dict[str, torch.Tensor],
                upper_graph: Optional[Dict[str, torch.Tensor]] = None,
                has_upper: bool = False) -> Dict[str, torch.Tensor]:
        """
        预测（带sigmoid）
        """
        results = self.forward(lower_graph, upper_graph, has_upper)
        
        # 对预测应用sigmoid
        results['predictions'] = torch.sigmoid(results['predictions'])
        results['lower_predictions'] = torch.sigmoid(results['lower_predictions'])
        
        return results


class UpperGraphEncoder(nn.Module):
    """
    上层图编码器（继承关系图）
    """
    
    def __init__(self,
                 input_dim: int = 64,
                 hidden_dim: int = 128,
                 output_dim: int = 128):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出嵌入维度
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 特征投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 简单图卷积
        self.conv1 = nn.Linear(hidden_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 池化
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, graph: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            graph: 上层图数据
                - node_features: [num_nodes, input_dim]
                - edge_index: [2, num_edges]
                - contract_to_idx: 合约到节点的映射
        
        Returns:
            图嵌入 [num_contracts, hidden_dim]
        """
        node_features = graph['node_features']
        edge_index = graph.get('edge_index')
        
        # 特征投影
        h = self.input_proj(node_features)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        
        # 第一层图卷积
        if edge_index is not None and edge_index.size(1) > 0:
            h = self._graph_conv(h, edge_index, self.conv1)
        
        # 第二层图卷积
        if edge_index is not None and edge_index.size(1) > 0:
            h = self._graph_conv(h, edge_index, self.conv2)
        
        # 池化
        num_nodes = node_features.size(0)
        
        # 全局池化：mean + max
        mean_pool = h.mean(dim=0, keepdim=True)
        max_pool = h.max(dim=0, keepdim=True)[0]
        
        graph_embedding = torch.cat([mean_pool, max_pool], dim=-1)
        graph_embedding = self.readout(graph_embedding)
        
        return graph_embedding
    
    def _graph_conv(self,
                    h: torch.Tensor,
                    edge_index: torch.Tensor,
                    conv: nn.Linear) -> torch.Tensor:
        """简单图卷积"""
        src, dst = edge_index
        
        # 邻居聚合
        neighbor_h = torch.zeros_like(h)
        neighbor_h.scatter_add_(0, dst.unsqueeze(-1).expand(-1, h.size(1)), h[src])
        
        # 消息传递
        out = conv(neighbor_h)
        out = F.relu(out)
        out = F.dropout(out, p=0.3, training=self.training)
        
        return out + h  # 残差连接


class DualLayerGraphClassifier(nn.Module):
    """
    双层图分类器（更简洁的版本）
    
    适用于：
    - 下层图为主，上层图为辅
    - 门控机制更简单
    - 图级别池化
    """
    
    def __init__(self,
                 node_feature_dim: int,
                 hidden_dim: int = 128,
                 num_classes: int = 4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 基线编码器
        self.baseline_encoder = GGNNEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_edge_types=8,
            propagation_rounds=2,
            propagation_substeps=20,
            dropout=0.3
        )
        
        # 上层图编码器（可选）
        self.upper_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 门控
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self,
                lower_features: torch.Tensor,
                lower_edges: torch.Tensor,
                lower_edge_types: torch.Tensor,
                upper_features: Optional[torch.Tensor] = None,
                upper_edges: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        """
        # 下层图编码
        lower_h = self.baseline_encoder(lower_features, lower_edges, lower_edge_types)
        
        # 图级别池化得到 [1, hidden_dim]
        lower_graph_h = lower_h.mean(dim=0, keepdim=True)
        
        # 上层图编码（如果提供）
        if upper_features is not None:
            upper_h = self.upper_encoder(upper_features)
            upper_graph_h = upper_h.mean(dim=0, keepdim=True)
        else:
            upper_graph_h = torch.zeros_like(lower_graph_h)
        
        # 门控融合 - 使用图级别嵌入
        combined = torch.cat([lower_graph_h, upper_graph_h], dim=-1)
        gate = self.gate(combined)
        fused = lower_graph_h + gate * upper_graph_h
        
        # 分类
        final_input = torch.cat([lower_graph_h, fused], dim=-1)
        logits = self.classifier(final_input)
        
        return {
            'logits': logits,
            'lower_embedding': lower_graph_h,
            'upper_embedding': upper_graph_h,
            'fused_embedding': fused,
            'gate': gate
        }


class MultiTaskDualLayerModel(nn.Module):
    """
    多任务双层图模型
    
    输出：
    1. 漏洞类型分类
    2. 风险等级评估
    3. 门控置信度
    """
    
    def __init__(self,
                 node_feature_dim: int,
                 hidden_dim: int = 128,
                 num_vuln_types: int = 4,
                 num_risk_levels: int = 3):
        super().__init__()
        
        # 双层图模型
        self.dual_model = DualLayerGNNModel(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_vuln_types,
            use_upper_graph=True
        )
        
        # 共享特征提取器
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 漏洞类型头
        self.vuln_head = nn.Linear(hidden_dim, num_vuln_types)
        
        # 风险等级头
        self.risk_head = nn.Linear(hidden_dim, num_risk_levels)
        
        # 置信度头
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
    def forward(self,
                lower_graph: Dict[str, torch.Tensor],
                upper_graph: Optional[Dict[str, torch.Tensor]] = None,
                has_upper: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        """
        # 双层图编码
        dual_results = self.dual_model(lower_graph, upper_graph, has_upper)
        
        # 共享特征
        shared = self.shared_fc(dual_results['fused_embedding'])
        
        # 多任务输出
        return {
            **dual_results,
            'vuln_logits': self.vuln_head(shared),
            'risk_logits': self.risk_head(shared),
            'confidence': self.confidence_head(shared)
        }


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    根据配置创建模型
    
    Args:
        config: 模型配置
    
    Returns:
        PyTorch模型
    """
    model_type = config.get('type', 'dual_layer')
    
    if model_type == 'dual_layer':
        return DualLayerGNNModel(
            node_feature_dim=config['node_feature_dim'],
            hidden_dim=config.get('hidden_dim', 128),
            num_classes=config.get('num_classes', 4),
            num_edge_types=config.get('num_edge_types', 8),
            use_upper_graph=config.get('use_upper_graph', True)
        )
    elif model_type == 'simple':
        return DualLayerGraphClassifier(
            node_feature_dim=config['node_feature_dim'],
            hidden_dim=config.get('hidden_dim', 128),
            num_classes=config.get('num_classes', 4)
        )
    elif model_type == 'multi_task':
        return MultiTaskDualLayerModel(
            node_feature_dim=config['node_feature_dim'],
            hidden_dim=config.get('hidden_dim', 128),
            num_vuln_types=config.get('num_classes', 4)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


