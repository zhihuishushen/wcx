"""
继承相关性预测器
预测漏洞是否与继承关系相关

这是核心创新模块：
- 在构建上层图之前，先预测漏洞是否与继承关系相关
- 如果相关 → 构建上层图
- 如果无关 → 跳过上层图，保持原版精度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class InheritanceFeatureExtractor(nn.Module):
    """
    继承相关性特征提取器
    从合约结构中提取与继承相关的特征
    """
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim) 继承相关特征
            
        Returns:
            (batch_size, hidden_dim // 2)
        """
        return self.feature_net(x)


class InheritancePredictor(nn.Module):
    """
    继承相关性预测器
    
    功能：
    1. 提取继承相关特征
    2. 预测漏洞与继承关系的相关性
    3. 输出置信度
    
    输入特征（8维）：
    - 继承深度 (inheritance_depth)
    - 函数数量 (num_functions)
    - 代码行数归一化 (num_lines / 1000)
    - 是否有delegatecall (has_delegatecall)
    - 是否有ether转账 (has_ether_transfer)
    - 是否有重入模式 (has_reentrancy_pattern)
    - 是否有call.value (has_call_value)
    - 是否有payable函数 (has_payable)
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        use_lower_graph_stats: bool = True,
        lower_graph_dim: int = 128,
    ):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            use_lower_graph_stats: 是否使用下层图统计特征
            lower_graph_dim: 下层图嵌入维度
        """
        super().__init__()
        
        self.use_lower_graph_stats = use_lower_graph_stats
        
        # 特征提取
        self.feature_extractor = InheritanceFeatureExtractor(input_dim, hidden_dim)
        
        # 如果使用下层图统计
        if use_lower_graph_stats:
            # 下层图投影
            self.lower_graph_proj = nn.Linear(lower_graph_dim, hidden_dim)
            
            # 融合层
            fusion_input_dim = hidden_dim // 2 + hidden_dim
        else:
            fusion_input_dim = hidden_dim // 2
        
        # 预测网络
        self.predictor = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # 置信度网络
        self.confidence_net = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        inheritance_features: torch.Tensor,
        lower_graph_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            inheritance_features: (batch_size, 8) 继承相关特征
            lower_graph_emb: (batch_size, lower_graph_dim) 下层图嵌入（可选）
            
        Returns:
            inheritance_prob: (batch_size, 1) 继承相关性概率
            confidence: (batch_size, 1) 预测置信度
        """
        # 提取特征
        feat = self.feature_extractor(inheritance_features)
        
        # 融合下层图信息
        if self.use_lower_graph_stats and lower_graph_emb is not None:
            lower_feat = self.lower_graph_proj(lower_graph_emb)
            fused = torch.cat([feat, lower_feat], dim=-1)
        else:
            fused = feat
        
        # 预测
        inheritance_prob = torch.sigmoid(self.predictor(fused))
        confidence = self.confidence_net(fused)
        
        return inheritance_prob, confidence
    
    def predict(
        self,
        inheritance_features: torch.Tensor,
        lower_graph_emb: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        预测并返回决策结果
        
        Args:
            inheritance_features: 继承相关特征
            lower_graph_emb: 下层图嵌入（可选）
            threshold: 决策阈值
            
        Returns:
            包含预测结果的字典:
            - inheritance_prob: 继承相关性概率
            - confidence: 置信度
            - use_upper_graph: 是否使用上层图 (bool)
            - decision: 'use_upper' 或 'lower_only'
        """
        inheritance_prob, confidence = self.forward(inheritance_features, lower_graph_emb)
        
        # 决策：基于概率和置信度
        use_upper = (inheritance_prob >= threshold).float()
        
        # 可选：基于置信度调整阈值
        # 如果置信度低，降低使用上层图的门槛
        adjusted_threshold = threshold * (1 - 0.2 * (1 - confidence))
        use_upper_adjusted = (inheritance_prob >= adjusted_threshold).float()
        
        return {
            'inheritance_prob': inheritance_prob,
            'confidence': confidence,
            'use_upper_graph': use_upper,
            'use_upper_adjusted': use_upper_adjusted,
        }


class TwoStageVulnerabilityModel(nn.Module):
    """
    两阶段漏洞检测模型
    
    阶段1: 继承相关性预测
    阶段2: 漏洞分类（基于下层图或双层图）
    """
    
    def __init__(
        self,
        # 下层图编码器参数
        node_feature_dim: int = 215,
        hidden_dim: int = 256,
        num_edge_types: int = 8,
        num_gnn_layers: int = 3,
        dropout: float = 0.1,
        
        # 继承相关性预测器参数
        inheritance_input_dim: int = 8,
        inheritance_hidden_dim: int = 64,
        
        # 模式选择
        mode: str = 'two_stage',  # 'two_stage', 'always', 'never', 'baseline'
    ):
        """
        Args:
            mode: 运行模式
                - 'two_stage': 两阶段（创新模式）
                - 'always': 始终使用上层图
                - 'never': 始终不使用上层图
                - 'baseline': 基线模式（只用下层图）
        """
        super().__init__()
        
        self.mode = mode
        
        # 下层图编码器
        from main.baseline_gnn.simple_gnn import BatchGGNNEncoder
        self.lower_encoder = BatchGGNNEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_edge_types=num_edge_types,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )
        
        # 继承相关性预测器
        self.inheritance_predictor = InheritancePredictor(
            input_dim=inheritance_input_dim,
            hidden_dim=inheritance_hidden_dim,
            use_lower_graph_stats=True,
            lower_graph_dim=hidden_dim,
        )
        
        # 上层图编码器（可选）
        self.upper_encoder = None
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 双层融合分类器
        self.fusion_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 门控（用于融合）
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_nodes: torch.Tensor,
        inheritance_features: torch.Tensor,
        upper_node_features: Optional[torch.Tensor] = None,
        upper_edge_index: Optional[torch.Tensor] = None,
        upper_edge_type: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            node_features: (batch_size, max_nodes, feature_dim) 下层图节点特征
            edge_index: (2, num_edges) 下层图边
            edge_type: (num_edges,) 下层图边类型
            num_nodes: (batch_size,) 每图节点数
            inheritance_features: (batch_size, 8) 继承相关特征
            upper_node_features: (batch_size, max_upper_nodes, feature_dim) 上层图节点特征
            upper_edge_index: (2, num_upper_edges) 上层图边
            upper_edge_type: (num_upper_edges,) 上层图边类型
            
        Returns:
            包含各种输出的字典
        """
        # 编码下层图
        lower_emb = self.lower_encoder(
            node_features=node_features,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes_per_graph=num_nodes,
        )
        
        # 阶段1: 继承相关性预测
        inheritance_prob, confidence = self.inheritance_predictor(
            inheritance_features, lower_emb
        )
        
        # 决定使用哪种模式
        if self.mode == 'always':
            use_upper = torch.ones_like(inheritance_prob)
        elif self.mode == 'never':
            use_upper = torch.zeros_like(inheritance_prob)
        elif self.mode == 'baseline':
            # 基线模式：始终只用下层图
            use_upper = torch.zeros_like(inheritance_prob)
        else:  # two_stage
            use_upper = (inheritance_prob >= 0.5).float()
        
        # 阶段2: 漏洞分类
        if use_upper.sum() > 0 and upper_node_features is not None:
            # 有样本需要使用上层图
            # 编码上层图
            upper_emb = self.upper_encoder(
                node_features=upper_node_features,
                edge_index=upper_edge_index,
                edge_type=upper_edge_type,
                num_nodes_per_graph=num_nodes,
            )
            
            # 融合
            combined = torch.cat([lower_emb, upper_emb], dim=-1)
            gate = self.fusion_gate(combined)
            fused_emb = gate * upper_emb + (1 - gate) * lower_emb
            
            # 分类
            logits = self.fusion_classifier(fused_emb)
        else:
            # 只用下层图
            logits = self.classifier(lower_emb)
        
        return {
            'logits': logits,
            'inheritance_prob': inheritance_prob,
            'confidence': confidence,
            'use_upper': use_upper,
            'lower_emb': lower_emb,
        }


def create_inheritance_label(
    contracts: list,
    vuln_type: str,
) -> torch.Tensor:
    """
    根据合约的继承关系创建继承相关性标签
    
    用于训练继承相关性预测器
    
    规则：
    - 如果合约有继承关系，且父合约有相同漏洞 → 1 (继承相关)
    - 否则 → 0 (继承无关)
    
    Args:
        contracts: 合约列表
        vuln_type: 漏洞类型
        
    Returns:
        继承相关性标签 (num_contracts,)
    """
    labels = []
    
    # 找出有漏洞的合约
    vulnerable_contracts = {c.file_id for c in contracts if c.label == 1}
    
    for contract in contracts:
        if contract.label == 0:
            # 无漏洞合约，标记为0
            labels.append(0)
        else:
            # 有漏洞合约
            if contract.inheritance_depth > 0 and contract.parent_contracts:
                # 有继承关系，可能与继承相关
                # 简化：检查父合约是否在有漏洞的合约中
                # 实际应用中需要更复杂的分析
                labels.append(1)
            else:
                # 无继承关系，与继承无关
                labels.append(0)
    
    return torch.tensor(labels, dtype=torch.float32)
