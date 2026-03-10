"""
损失函数模块
包含混合损失函数和Focal Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance
    
    特点：
    1. 关注困难样本
    2. 自动调整类别权重
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0,
                 reduction: str = 'mean', class_weights: torch.Tensor = None):
        """
        Args:
            alpha: 类别权重因子
            gamma: 聚焦因子
            reduction: 归约方式 ('mean', 'sum', 'none')
            class_weights: 类别权重 [num_classes]
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 预测概率 [batch, num_classes]
            targets: 真实标签 [batch, num_classes]
        
        Returns:
            loss: Focal Loss
        """
        # 计算二元交叉熵
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # 计算pt
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        
        # 计算Focal Loss
        focal_weight = (1 - pt) ** self.gamma
        focal_weight = torch.where(targets == 1, self.alpha * focal_weight, focal_weight)
        
        # 应用类别权重
        if self.class_weights is not None:
            weights = self.class_weights.to(inputs.device)
            focal_weight = focal_weight * weights.unsqueeze(0)
        
        loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class HybridLoss(nn.Module):
    """
    混合损失函数
    
    结合：
    1. 基础分类损失（BCE）
    2. 伪标签损失
    3. 置信度加权损失
    """
    
    def __init__(self, 
                 baseline_weight: float = 1.0,
                 pseudo_weight: float = 0.5,
                 confidence_weight: float = 0.1,
                 focal_gamma: float = 2.0,
                 use_focal: bool = True,
                 class_weights: torch.Tensor = None):
        """
        Args:
            baseline_weight: 基础损失权重
            pseudo_weight: 伪标签损失权重
            confidence_weight: 置信度损失权重
            focal_gamma: Focal Loss的gamma参数
            use_focal: 是否使用Focal Loss
            class_weights: 类别权重 [num_classes]
        """
        super().__init__()
        
        self.baseline_weight = baseline_weight
        self.pseudo_weight = pseudo_weight
        self.confidence_weight = confidence_weight
        
        self.use_focal = use_focal
        self.class_weights = class_weights
        self.focal = FocalLoss(gamma=focal_gamma, class_weights=class_weights)
        
        # BCE损失
        self.bce = nn.BCELoss()
    
    def set_class_weights(self, class_weights: torch.Tensor):
        """动态设置类别权重"""
        self.class_weights = class_weights
        self.focal.class_weights = class_weights
    
    def forward(self, 
                predictions: torch.Tensor,
                targets: torch.Tensor,
                pseudo_labels: Optional[torch.Tensor] = None,
                pseudo_confidence: Optional[torch.Tensor] = None,
                baseline_weight: Optional[float] = None,
                pseudo_weight: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: 预测概率 [batch, num_classes]
            targets: 真实标签 [batch, num_classes]
            pseudo_labels: 伪标签 [batch, num_classes]
            pseudo_confidence: 伪标签置信度 [batch, 1]
            baseline_weight: 动态基础损失权重
            pseudo_weight: 动态伪标签损失权重
        
        Returns:
            {
                'total_loss': 总损失,
                'baseline_loss': 基础损失,
                'pseudo_loss': 伪标签损失,
                'confidence_loss': 置信度损失
            }
        """
        # 动态权重
        bw = baseline_weight if baseline_weight is not None else self.baseline_weight
        pw = pseudo_weight if pseudo_weight is not None else self.pseudo_weight
        
        # 1. 基础分类损失
        if self.use_focal:
            baseline_loss = self.focal(predictions, targets)
        else:
            baseline_loss = self.bce(predictions, targets)
        
        total_loss = bw * baseline_loss
        
        # 2. 伪标签损失（仅对伪标签样本计算）
        pseudo_loss = torch.tensor(0.0, device=predictions.device)
        if pseudo_labels is not None and pseudo_confidence is not None:
            # 找出有伪标签的样本
            has_pseudo = pseudo_labels.sum(dim=-1) > 0
            
            if has_pseudo.any():
                # 仅对有伪标签的样本计算损失
                pseudo_mask = has_pseudo.unsqueeze(-1).float()
                
                # 置信度加权
                pseudo_pred = predictions * pseudo_mask
                pseudo_tgt = pseudo_labels * pseudo_mask
                
                if self.use_focal:
                    pseudo_loss = self.focal(pseudo_pred, pseudo_tgt)
                else:
                    pseudo_loss = self.bce(pseudo_pred, pseudo_tgt)
                
                # 应用置信度权重
                confidence_weight = pseudo_confidence.mean()
                pseudo_loss = pseudo_loss * confidence_weight
                
                total_loss = total_loss + pw * pseudo_loss
        
        # 3. 置信度正则化损失（可选）
        confidence_loss = torch.tensor(0.0, device=predictions.device)
        if pseudo_confidence is not None:
            # 鼓励置信度与预测准确性一致
            # 当预测接近0或1时，置信度应该高
            prediction_entropy = -(predictions * torch.log(predictions + 1e-8)).sum(dim=-1)
            confidence_loss = (pseudo_confidence.squeeze() * prediction_entropy).mean()
            
            total_loss = total_loss + self.confidence_weight * confidence_loss
        
        return {
            'total_loss': total_loss,
            'baseline_loss': baseline_loss,
            'pseudo_loss': pseudo_loss,
            'confidence_loss': confidence_loss
        }


class GateLoss(nn.Module):
    """
    门控损失函数
    
    功能：
    1. 门控信号正则化
    2. 门控决策与真实需求的匹配
    """
    
    def __init__(self, gate_regularization: float = 0.01):
        """
        Args:
            gate_regularization: 门控正则化强度
        """
        super().__init__()
        self.gate_regularization = gate_regularization
    
    def forward(self, 
                use_upper: torch.Tensor,
                should_use_upper: torch.Tensor,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            use_upper: 门控预测 [batch, 1]
            should_use_upper: 真实门控需求 [batch, 1]
            predictions: 预测概率 [batch, num_classes]
            targets: 真实标签 [batch, num_classes]
        
        Returns:
            {
                'gate_loss': 门控损失,
                'gate_accuracy': 门控准确率,
                'regularization': 正则化项
            }
        """
        # 门控二元交叉熵
        gate_loss = F.binary_cross_entropy(use_upper, should_use_upper.float())
        
        # 门控准确率
        gate_pred = (use_upper > 0.5).float()
        gate_accuracy = (gate_pred == should_use_upper.float()).float().mean()
        
        # 正则化：避免门控过于频繁或过于稀少
        gate_mean = use_upper.mean()
        regularization = self.gate_regularization * (
            torch.abs(gate_mean - 0.5)  # 鼓励门控使用率在50%左右
        )
        
        total_loss = gate_loss + regularization
        
        return {
            'gate_loss': gate_loss,
            'gate_accuracy': gate_accuracy,
            'regularization': regularization,
            'total_loss': total_loss,
            'gate_mean': gate_mean
        }


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    
    结合分类损失和门控损失
    """
    
    def __init__(self, classification_weight: float = 1.0,
                 gate_weight: float = 0.1):
        """
        Args:
            classification_weight: 分类损失权重
            gate_weight: 门控损失权重
        """
        super().__init__()
        
        self.classification_weight = classification_weight
        self.gate_weight = gate_weight
        
        self.classification_loss = HybridLoss()
        self.gate_loss = GateLoss()
    
    def forward(self, 
                predictions: torch.Tensor,
                targets: torch.Tensor,
                use_upper: torch.Tensor,
                should_use_upper: torch.Tensor,
                pseudo_labels: torch.Tensor = None,
                pseudo_confidence: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: 预测概率 [batch, num_classes]
            targets: 真实标签 [batch, num_classes]
            use_upper: 门控预测 [batch, 1]
            should_use_upper: 真实门控需求 [batch, 1]
            pseudo_labels: 伪标签 [batch, num_classes]
            pseudo_confidence: 伪标签置信度 [batch, 1]
        
        Returns:
            多任务损失字典
        """
        # 分类损失
        class_loss = self.classification_loss(
            predictions, targets, pseudo_labels, pseudo_confidence
        )
        
        # 门控损失
        gate_loss = self.gate_loss(use_upper, should_use_upper, predictions, targets)
        
        # 总损失
        total_loss = (
            self.classification_weight * class_loss['total_loss'] +
            self.gate_weight * gate_loss['total_loss']
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': class_loss['total_loss'],
            'baseline_loss': class_loss['baseline_loss'],
            'pseudo_loss': class_loss['pseudo_loss'],
            'gate_loss': gate_loss['total_loss'],
            'gate_accuracy': gate_loss['gate_accuracy'],
            'class_loss_dict': class_loss,
            'gate_loss_dict': gate_loss
        }

