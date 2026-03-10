"""
置信度计算模块
评估门控决策和预测结果的可靠性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class ConfidenceCalculator(nn.Module):
    """
    置信度计算器
    
    功能：
    1. 评估门控决策的可靠性
    2. 计算预测置信度
    3. 识别困难样本
    """
    
    def __init__(self, graph_dim: int = 128, inheritance_dim: int = 64, hidden_dim: int = 128):
        """
        Args:
            graph_dim: 图特征维度
            inheritance_dim: 继承关系特征维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.graph_dim = graph_dim
        self.inheritance_dim = inheritance_dim
        
        fusion_dim = graph_dim + inheritance_dim
        
        # 置信度评估网络
        self.confidence_net = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 不确定性估计网络
        self.uncertainty_net = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, graph_emb: torch.Tensor,
                inheritance_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        计算置信度和不确定性
        
        Args:
            graph_emb: 图特征 [batch, hidden_dim]
            inheritance_emb: 继承关系特征 [batch, hidden_dim]
        
        Returns:
            confidence: 置信度 [batch, 1]
            uncertainty: 不确定性 [batch, 1]
            info: 调试信息
        """
        # 融合特征
        fused = torch.cat([graph_emb, inheritance_emb], dim=-1)
        
        # 计算置信度
        confidence = self.confidence_net(fused)
        
        # 计算不确定性
        uncertainty = self.uncertainty_net(fused)
        
        # 调试信息
        info = {
            'confidence_mean': confidence.mean().item(),
            'confidence_std': confidence.std().item(),
            'uncertainty_mean': uncertainty.mean().item(),
            'uncertainty_std': uncertainty.std().item(),
        }
        
        return confidence, uncertainty, info
    
    def compute_prediction_confidence(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        计算预测置信度（基于预测概率）
        
        Args:
            predictions: 预测概率 [batch, num_classes]
        
        Returns:
            置信度 [batch, 1]
        """
        # 最大概率作为置信度
        max_prob = predictions.max(dim=-1)[0]  # [batch]
        
        # 概率分布的熵（越低置信度越高）
        entropy = -(predictions * torch.log(predictions + 1e-8)).sum(dim=-1)  # [batch]
        
        # 归一化熵到[0,1]
        max_entropy = np.log(predictions.size(-1))
        normalized_entropy = 1 - entropy / (max_entropy + 1e-8)
        
        # 综合置信度
        confidence = 0.5 * max_prob + 0.5 * normalized_entropy
        
        return confidence.unsqueeze(-1)
    
    def compute_gate_confidence(self, use_upper: torch.Tensor,
                               inheritance_features: Dict) -> torch.Tensor:
        """
        计算门控置信度
        
        Args:
            use_upper: 门控信号 [batch, 1]
            inheritance_features: 继承关系特征字典
        
        Returns:
            门控置信度 [batch, 1]
        """
        # 基于继承关系特征的置信度
        if 'suspicious_scores' in inheritance_features:
            scores = inheritance_features['suspicious_scores']
            if isinstance(scores, torch.Tensor):
                score_confidence = scores.mean(dim=-1, keepdim=True)
            else:
                score_confidence = torch.tensor(
                    [[np.mean(list(scores.values()))]] * use_upper.size(0),
                    device=use_upper.device
                )
        else:
            score_confidence = use_upper
        
        # 基于继承关系复杂度的置信度
        if 'complexity' in inheritance_features:
            complexity = inheritance_features['complexity']
            if isinstance(complexity, torch.Tensor):
                complexity_confidence = 1.0 / (complexity + 1.0)
            else:
                complexity_confidence = torch.tensor(
                    [[1.0 / (complexity + 1.0)]] * use_upper.size(0),
                    device=use_upper.device
                )
        else:
            complexity_confidence = use_upper.clone()
        
        # 综合置信度
        confidence = 0.6 * score_confidence + 0.4 * complexity_confidence
        confidence = torch.clamp(confidence, 0.0, 1.0)
        
        return confidence
    
    def identify_hard_samples(self, confidence: torch.Tensor,
                             predictions: torch.Tensor,
                             threshold: float = 0.5) -> torch.Tensor:
        """
        识别困难样本
        
        Args:
            confidence: 置信度 [batch, 1]
            predictions: 预测概率 [batch, num_classes]
            threshold: 困难样本阈值
        
        Returns:
            困难样本掩码 [batch]
        """
        # 低置信度样本
        low_confidence = confidence.squeeze(-1) < threshold
        
        # 预测概率接近0.5的样本（不确定）
        entropy = -(predictions * torch.log(predictions + 1e-8)).sum(dim=-1)
        high_entropy = entropy > np.log(2)
        
        # 综合判断
        hard_samples = low_confidence | high_entropy
        
        return hard_samples


class EnsembleConfidence(nn.Module):
    """
    集成置信度计算器
    
    使用多个模型或多次预测来估计置信度
    """
    
    def __init__(self, base_calculator: ConfidenceCalculator, num_samples: int = 10):
        """
        Args:
            base_calculator: 基础置信度计算器
            num_samples: MC Dropout采样次数
        """
        super().__init__()
        self.base_calculator = base_calculator
        self.num_samples = num_samples
    
    def forward_with_uncertainty(self, model: nn.Module,
                                  graph_emb: torch.Tensor,
                                  inheritance_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        使用MC Dropout估计不确定性
        
        Args:
            model: 带有Dropout的模型
            graph_emb: 图特征
            inheritance_emb: 继承关系特征
        
        Returns:
            mean_pred: 平均预测
            uncertainty: 不确定性
            info: 调试信息
        """
        model.eval()
        
        predictions = []
        for _ in range(self.num_samples):
            with torch.enable_grad():
                pred = model(graph_emb, inheritance_emb)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch, num_classes]
        
        # 计算均值和方差
        mean_pred = predictions.mean(dim=0)  # [batch, num_classes]
        uncertainty = predictions.var(dim=0)  # [batch, num_classes]
        
        # 汇总不确定性
        uncertainty_scalar = uncertainty.mean(dim=-1)  # [batch]
        
        info = {
            'prediction_std': uncertainty_scalar.mean().item(),
            'prediction_max_std': uncertainty_scalar.max().item()
        }
        
        return mean_pred, uncertainty_scalar.unsqueeze(-1), info

