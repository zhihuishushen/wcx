"""
双层图融合模块
实现目标图和上层图的融合预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class PredictionFusion(nn.Module):
    """
    预测结果融合层
    
    功能：
    1. 基础预测 + 增强预测加权融合
    2. 融合权重 = 置信度
    3. 低置信度时退化为基础预测
    """
    
    def __init__(self, hidden_dim: int = 128):
        """
        Args:
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def fuse(self, baseline_pred: torch.Tensor,
             enhanced_pred: torch.Tensor,
             confidence: torch.Tensor) -> torch.Tensor:
        """
        融合预测结果
        
        Args:
            baseline_pred: 原版预测 [batch, 1] 或 [batch, num_classes]
            enhanced_pred: 增强预测 [batch, 1] 或 [batch, num_classes]
            confidence: 置信度 [batch, 1]
        
        Returns:
            融合预测 [batch, 1] 或 [batch, num_classes]
        """
        # 确保维度一致
        if baseline_pred.dim() == 1:
            baseline_pred = baseline_pred.unsqueeze(-1)
        if enhanced_pred.dim() == 1:
            enhanced_pred = enhanced_pred.unsqueeze(-1)
        
        # 置信度权重
        weight = confidence.unsqueeze(-1)  # [batch, 1, 1]
        
        # 加权融合
        fused = (1 - weight) * baseline_pred + weight * enhanced_pred
        
        return fused
    
    def forward(self, baseline_pred: torch.Tensor,
                enhanced_pred: torch.Tensor,
                confidence: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            baseline_pred: 原版预测
            enhanced_pred: 增强预测
            confidence: 置信度
        
        Returns:
            融合预测
        """
        return self.fuse(baseline_pred, enhanced_pred, confidence)


class DualGraphFusion(nn.Module):
    """
    双层图融合网络
    
    功能：
    1. 分别编码目标图和上层图
    2. 通过注意力机制融合双层信息
    3. 输出最终预测
    """
    
    def __init__(self, node_emb_dim: int, hidden_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2):
        """
        Args:
            node_emb_dim: 节点嵌入维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            num_layers: 融合层数
        """
        super().__init__()
        
        self.node_emb_dim = node_emb_dim
        self.hidden_dim = hidden_dim
        
        # 目标图投影
        self.target_proj = nn.Linear(node_emb_dim, hidden_dim)
        
        # 上层图投影
        self.upper_proj = nn.Linear(node_emb_dim, hidden_dim)
        
        # 跨层注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 融合层 - 只有第一层需要处理拼接后的维度
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),  # 第一层: 256 -> 128
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        ])
        # 后续层（如果 num_layers > 1）使用 hidden_dim
        for _ in range(1, num_layers):
            self.fusion_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, target_emb: torch.Tensor,
                upper_emb: torch.Tensor,
                upper_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            target_emb: 目标图节点嵌入 [num_target_nodes, node_emb_dim]
            upper_emb: 上层图节点嵌入 [num_upper_nodes, node_emb_dim]
            upper_mask: 上层图掩码 [num_upper_nodes]
        
        Returns:
            fused_pred: 融合预测 [1]
            info: 调试信息
        """
        # 投影
        target_feat = self.target_proj(target_emb)  # [num_target, hidden_dim]
        upper_feat = self.upper_proj(upper_emb)  # [num_upper, hidden_dim]
        
        # 全局池化 - 确保形状始终是 [1, hidden_dim]
        if target_feat.dim() == 2:
            target_global = target_feat.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        else:
            target_global = target_feat
        
        if upper_emb.size(0) > 0:
            # 跨层注意力
            query = target_global  # [1, 128]
            key = upper_feat  # [num_upper, 128]
            
            # 计算注意力分数
            attention_scores = torch.matmul(query, key.transpose(0, 1)) / (self.hidden_dim ** 0.5)
            attention_weights = torch.softmax(attention_scores, dim=-1)  # [1, num_upper]
            attended_upper = torch.matmul(attention_weights, upper_feat)  # [1, hidden_dim]
            
            # 融合：拼接 [1, 128] + [1, 128] = [1, 256]
            fused = torch.cat([target_global, attended_upper], dim=-1)
            
            # 通过融合层
            for fusion_layer in self.fusion_layers:
                fused = fusion_layer(fused)
            
            # 预测
            pred = self.predictor(fused)  # [1, 1]
            
            info = {
                'target_global_norm': target_global.norm().item(),
                'upper_global_norm': attended_upper.norm().item(),
                'attn_weights_mean': attention_weights.mean().item()
            }
        else:
            # 无上层图时直接使用目标图
            pred = self.predictor(target_global)
            
            info = {
                'target_global_norm': target_global.norm().item(),
                'upper_global_norm': 0.0,
                'attn_weights_mean': 0.0
            }
        
        return pred.squeeze(0), info
    
    def apply_with_gate(self, target_emb: torch.Tensor,
                       upper_emb: torch.Tensor,
                       gate: torch.Tensor,
                       confidence: torch.Tensor,
                       upper_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        使用门控信号融合
        
        Args:
            target_emb: 目标图嵌入
            upper_emb: 上层图嵌入
            gate: 门控信号 [batch, 1]
            confidence: 置信度 [batch, 1]
            upper_mask: 上层图掩码
        
        Returns:
            融合预测
        """
        # 计算增强预测
        enhanced_pred, info = self.forward(target_emb, upper_emb, upper_mask)
        
        # 基础预测（仅使用目标图）
        target_global = self.target_proj(target_emb).mean(dim=0, keepdim=True)
        baseline_pred = self.predictor(target_global)
        
        # 加权融合
        weight = gate * confidence
        weight = torch.clamp(weight, 0.0, 1.0)
        
        fused_pred = (1 - weight) * baseline_pred + weight * enhanced_pred.unsqueeze(0)
        
        info['gate'] = gate.item()
        info['confidence'] = confidence.item()
        
        return fused_pred.squeeze(0), info


class HierarchicalFusion(nn.Module):
    """
    层次化融合网络
    
    特点：
    1. 多层次融合（节点级 -> 图级）
    2. 支持条件融合（根据门控信号）
    3. 残差连接
    """
    
    def __init__(self, node_emb_dim: int, graph_emb_dim: int,
                 hidden_dim: int = 128, num_heads: int = 4):
        """
        Args:
            node_emb_dim: 节点嵌入维度
            graph_emb_dim: 图嵌入维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
        """
        super().__init__()
        
        self.node_emb_dim = node_emb_dim
        self.graph_emb_dim = graph_emb_dim
        
        # 节点级融合
        self.node_fusion = nn.MultiheadAttention(
            embed_dim=node_emb_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 图级融合
        self.graph_fusion = nn.Sequential(
            nn.Linear(graph_emb_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 门控融合
        self.gate_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 残差投影
        self.residual = nn.Linear(graph_emb_dim, hidden_dim)
    
    def forward(self, target_nodes: torch.Tensor,
                upper_nodes: torch.Tensor,
                target_graph_emb: torch.Tensor,
                upper_graph_emb: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            target_nodes: 目标图节点 [num_target, node_emb_dim]
            upper_nodes: 上层图节点 [num_upper, node_emb_dim]
            target_graph_emb: 目标图嵌入 [graph_emb_dim]
            upper_graph_emb: 上层图嵌入 [graph_emb_dim]
        
        Returns:
            融合嵌入
        """
        batch_size = target_nodes.size(0)
        
        # 节点级融合
        if upper_nodes.size(0) > 0:
            # 注意力融合
            query = target_nodes.unsqueeze(0)  # [1, num_target, dim]
            key = upper_nodes.unsqueeze(0)
            value = upper_nodes.unsqueeze(0)
            
            fused_nodes, node_attn = self.node_fusion(query, key, value)
            fused_nodes = fused_nodes.squeeze(0)  # [num_target, dim]
        else:
            fused_nodes = target_nodes
            node_attn = torch.tensor(0.0)
        
        # 图级融合
        if upper_graph_emb.numel() > 0:
            # 融合图嵌入
            graph_fused = self.graph_fusion(
                torch.cat([target_graph_emb, upper_graph_emb], dim=-1)
            )
        else:
            graph_fused = self.residual(target_graph_emb)
        
        # 门控加权
        gate_input = torch.cat([
            target_graph_emb.unsqueeze(0),
            upper_graph_emb.unsqueeze(0)
        ], dim=-1)
        gate = self.gate_fusion(gate_input)  # [1, 1]
        
        # 应用门控
        weighted = gate * graph_fused + (1 - gate) * self.residual(target_graph_emb)
        
        info = {
            'node_attention': node_attn.item() if isinstance(node_attn, torch.Tensor) else node_attn,
            'gate_value': gate.item(),
            'graph_fused_norm': graph_fused.norm().item()
        }
        
        return weighted, info

