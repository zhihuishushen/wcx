"""
可学习门控模块
动态决定是否使用上层图信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class LearnableGate(nn.Module):
    """
    可学习门控模块
    
    功能：
    1. 融合图特征和继承关系特征
    2. 输出门控信号（是否使用上层图）
    3. 输出置信度分数
    
    特点：
    - 使用可学习的神经网络
    - 支持多种融合策略
    - 输出置信度评估决策可靠性
    """
    
    def __init__(self, graph_dim: int, inheritance_dim: int, hidden_dim: int = 64):
        """
        Args:
            graph_dim: 图特征维度
            inheritance_dim: 继承关系特征维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.graph_dim = graph_dim
        self.inheritance_dim = inheritance_dim

        # 特征投影层（可选）- 如果维度不匹配，先投影再拼接
        self.use_projection = graph_dim != hidden_dim or inheritance_dim != hidden_dim
        if self.use_projection:
            self.graph_projector = nn.Linear(graph_dim, hidden_dim)
            self.inheritance_projector = nn.Linear(inheritance_dim, hidden_dim)
            fusion_input_dim = hidden_dim * 2
        else:
            self.graph_projector = None
            self.inheritance_projector = None
            fusion_input_dim = graph_dim + inheritance_dim

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 门控网络（决定是否使用上层图）
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 置信度网络（评估决策的可靠性）
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 特征投影层（可选）
        self.graph_projector = nn.Linear(graph_dim, hidden_dim) if graph_dim != hidden_dim else None
        self.inheritance_projector = nn.Linear(inheritance_dim, hidden_dim) if inheritance_dim != hidden_dim else None
    
    def forward(self, graph_emb: torch.Tensor, 
                inheritance_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            graph_emb: 目标图编码 [batch, graph_dim]
            inheritance_emb: 继承关系特征 [batch, inheritance_dim]
        
        Returns:
            use_upper: 是否使用上层图 [batch, 1]
            confidence: 决策置信度 [batch, 1]
        """
        # 特征投影
        if self.graph_projector is not None:
            graph_emb = self.graph_projector(graph_emb)
        if self.inheritance_projector is not None:
            inheritance_emb = self.inheritance_projector(inheritance_emb)
        
        # 融合特征
        fused = self.fusion(torch.cat([graph_emb, inheritance_emb], dim=-1))
        
        # 门控决策
        use_upper = self.gate_net(fused)
        
        # 置信度评估
        confidence = self.confidence_net(fused)
        
        return use_upper, confidence


class AdaptiveGate(nn.Module):
    """
    自适应门控模块（增强版）
    
    核心设计：门控仅基于下层图embedding预测是否需要上层图信息
    这样可以在不知道上层图内容的情况下做出决策
    
    增强功能：
    1. 仅使用下层图特征做决策
    2. 多尺度特征提取
    3. 温度参数控制软/硬切换
    """
    
    def __init__(self, graph_dim: int, inheritance_dim: int = None, 
                 hidden_dim: int = 128, num_heads: int = 4):
        """
        Args:
            graph_dim: 图特征维度（下层图）
            inheritance_dim: 继承关系特征维度（可选，用于辅助预测）
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
        """
        super().__init__()
        
        self.graph_dim = graph_dim
        self.inheritance_dim = inheritance_dim
        self.hidden_dim = hidden_dim
        self.has_inheritance = inheritance_dim is not None
        
        # 温度参数（用于调节门控的锐度）
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # 下层图特征投影
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        
        # 如果有继承关系特征，也投影（但不直接用于门控决策）
        if self.has_inheritance:
        self.inheritance_proj = nn.Linear(inheritance_dim, hidden_dim)
        
        # 下层图多尺度特征提取
        self.scale_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, hidden_dim)
            )
            for _ in range(3)
        ])
        
        # 注意力机制用于处理下层图
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 门控网络 - 仅基于下层图特征
        # 输入维度: hidden_dim * 2 (graph_feat + graph_attn)
        gate_input_dim = hidden_dim * 2
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 置信度网络 - 评估决策的可靠性
        self.confidence_net = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 特征变换（用于门控加权融合）
        self.graph_transform = nn.Linear(hidden_dim, hidden_dim)
        
        # 如果有继承关系特征，用于融合
        if self.has_inheritance:
        self.inheritance_transform = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, graph_emb: torch.Tensor,
                inheritance_emb: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播 - 门控仅基于下层图embedding
        
        Args:
            graph_emb: 下层图编码 [batch, graph_dim]
            inheritance_emb: 继承关系特征（可选）[batch, inheritance_dim]
        
        Returns:
            use_upper: 是否使用上层图 [batch, 1]
            confidence: 决策置信度 [batch, 1]
            debug_info: 调试信息
        """
        batch_size = graph_emb.size(0)
        
        # 下层图特征投影
        graph_feat = self.graph_proj(graph_emb)  # [batch, hidden_dim]
        
        # 多尺度融合
        graph_multi = torch.stack([sf(graph_feat) for sf in self.scale_fusion], dim=1)  # [batch, 3, hidden_dim]
        
        # 注意力融合
        graph_attn, _ = self.attention(
            graph_feat.unsqueeze(1),
            graph_multi,
            graph_multi
        )  # [batch, 1, hidden_dim]
        graph_attn = graph_attn.squeeze(1)  # [batch, hidden_dim]
        
        # 拼接下层图特征
        graph_fused = torch.cat([graph_feat, graph_attn], dim=-1)  # [batch, hidden_dim * 2]
        
        # 门控决策 - 仅基于下层图
        use_upper = self.gate_net(graph_fused)
        
        # 温度调节 (可选)
        # use_upper = torch.pow(use_upper, 1.0 / self.temperature)
        
        # 限制范围避免极端值
        use_upper = torch.clamp(use_upper, min=0.05, max=0.95)
        
        # 置信度评估
        confidence = self.confidence_net(graph_fused)
        
        # 调试信息
        debug_info = {
            'graph_fused_norm': graph_feat.norm(dim=-1).mean().item(),
            'gate_mean': use_upper.mean().item(),
            'confidence_mean': confidence.mean().item(),
            'temperature': self.temperature.item()
        }
        
        return use_upper, confidence, debug_info
    
    def forward_with_inheritance(self, graph_emb: torch.Tensor,
                                inheritance_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        带继承关系特征的融合预测（用于最终预测）
        
        注意：门控决策仍然只基于下层图
        """
        # 门控决策
        use_upper, confidence, debug_info = self.forward(graph_emb)
        
        # 融合特征（用于最终预测）
        if self.has_inheritance:
            graph_feat = self.graph_proj(graph_emb)
            inheritance_feat = self.inheritance_proj(inheritance_emb)
            
            # 门控加权融合
            graph_trans = self.graph_transform(graph_feat)
            inherit_trans = self.inheritance_transform(inheritance_feat)
            
            # 加权融合
            fused = graph_trans + use_upper * inherit_trans
            debug_info['fused_norm'] = fused.norm(dim=-1).mean().item()
        else:
            fused = self.graph_transform(self.graph_proj(graph_emb))
        
        return use_upper, confidence, fused, debug_info
    
    def apply_gating(self, lower_pred: torch.Tensor,
                    upper_pred: torch.Tensor,
                    use_upper: torch.Tensor) -> torch.Tensor:
        """
        应用门控融合预测结果
        
        策略：
        - use_upper * upper_pred + (1 - use_upper) * lower_pred
        
        Args:
            lower_pred: 下层图预测 [batch, num_classes]
            upper_pred: 上层图预测 [batch, num_classes]
            use_upper: 门控信号 [batch, 1]
        
        Returns:
            融合预测 [batch, num_classes]
        """
        # 确保维度匹配
        if use_upper.dim() == 1:
            use_upper = use_upper.unsqueeze(-1)
        
        # 门控加权融合
        fused = (1 - use_upper) * lower_pred + use_upper * upper_pred
        
        return fused


class FixedGate(nn.Module):
    """
    固定阈值门控（基线对比）
    
    使用固定阈值决定是否使用上层图
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: 阈值
        """
        super().__init__()
        self.threshold = threshold
    
    def forward(self, graph_emb: torch.Tensor,
                inheritance_emb: torch.Tensor,
                suspicious_score: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            graph_emb: 目标图编码 [batch, graph_dim]
            inheritance_emb: 继承关系特征 [batch, inheritance_dim]
            suspicious_score: 可疑分数（可选）
        
        Returns:
            use_upper: 是否使用上层图 [batch, 1]
            confidence: 置信度 [batch, 1]
        """
        batch_size = graph_emb.size(0)
        
        # 使用可疑分数或默认值
        if suspicious_score is not None:
            use_upper = torch.tensor([[suspicious_score]] * batch_size, device=graph_emb.device)
        else:
            use_upper = torch.full((batch_size, 1), self.threshold, device=graph_emb.device)
        
        # 置信度与门控值相同
        confidence = use_upper.clone()
        
        return use_upper, confidence

