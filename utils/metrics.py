"""
评估指标模块
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)


def compute_metrics(predictions: torch.Tensor, 
                   targets: torch.Tensor,
                   thresholds: Optional[Dict[str, float]] = None) -> Dict:
    """
    计算评估指标
    
    Args:
        predictions: 预测概率 [batch, num_classes]
        targets: 真实标签 [batch, num_classes]
        thresholds: 每个类型的阈值
    
    Returns:
        指标字典
    """
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    # 默认阈值
    if thresholds is None:
        thresholds = {i: 0.5 for i in range(predictions_np.shape[1])}
    
    # 二值化预测
    binary_preds = np.zeros_like(predictions_np)
    for i in range(predictions_np.shape[1]):
        binary_preds[:, i] = (predictions_np[:, i] >= thresholds.get(i, 0.5)).astype(float)
    
    metrics = {}
    
    # 整体准确率
    metrics['accuracy'] = accuracy_score(targets_np.flatten(), binary_preds.flatten())
    
    # 宏平均
    metrics['macro_precision'] = precision_score(targets_np, binary_preds, average='macro', zero_division=0)
    metrics['macro_recall'] = recall_score(targets_np, binary_preds, average='macro', zero_division=0)
    metrics['macro_f1'] = f1_score(targets_np, binary_preds, average='macro', zero_division=0)
    
    # 加权平均
    metrics['weighted_precision'] = precision_score(targets_np, binary_preds, average='weighted', zero_division=0)
    metrics['weighted_recall'] = recall_score(targets_np, binary_preds, average='weighted', zero_division=0)
    metrics['weighted_f1'] = f1_score(targets_np, binary_preds, average='weighted', zero_division=0)
    
    # 每类指标
    per_class_metrics = {}
    for i in range(predictions_np.shape[1]):
        tp = ((binary_preds[:, i] == 1) & (targets_np[:, i] == 1)).sum()
        fp = ((binary_preds[:, i] == 1) & (targets_np[:, i] == 0)).sum()
        tn = ((binary_preds[:, i] == 0) & (targets_np[:, i] == 0)).sum()
        fn = ((binary_preds[:, i] == 0) & (targets_np[:, i] == 1)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # AUC
        if len(np.unique(targets_np[:, i])) > 1:
            auc = roc_auc_score(targets_np[:, i], predictions_np[:, i])
        else:
            auc = 0.5
        
        per_class_metrics[i] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
    
    metrics['per_class'] = per_class_metrics
    
    # AUC
    try:
        metrics['macro_auc'] = roc_auc_score(targets_np, predictions_np, average='macro')
        metrics['weighted_auc'] = roc_auc_score(targets_np, predictions_np, average='weighted')
    except:
        metrics['macro_auc'] = 0.5
        metrics['weighted_auc'] = 0.5
    
    return metrics


def compute_confusion_matrix(predictions: torch.Tensor,
                            targets: torch.Tensor,
                            num_classes: int = 2) -> np.ndarray:
    """
    计算混淆矩阵
    
    Args:
        predictions: 预测 [batch, num_classes] 或 [batch]
        targets: 目标 [batch, num_classes] 或 [batch]
        num_classes: 类别数
    
    Returns:
        混淆矩阵
    """
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    # 转换为二值
    if predictions_np.ndim > 1:
        predictions_np = (predictions_np > 0.5).astype(int).flatten()
    else:
        predictions_np = (predictions_np > 0.5).astype(int)
    
    if targets_np.ndim > 1:
        targets_np = targets_np.astype(int).flatten()
    
    return confusion_matrix(targets_np, predictions_np, labels=list(range(num_classes)))


def find_optimal_thresholds(predictions: torch.Tensor,
                           targets: torch.Tensor,
                           method: str = 'f1') -> Dict[int, float]:
    """
    寻找最优阈值
    
    Args:
        predictions: 预测概率
        targets: 真实标签
        method: 优化方法 ('f1', 'precision', 'recall')
    
    Returns:
        每个类型的最优阈值
    """
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    optimal_thresholds = {}
    
    for i in range(predictions_np.shape[1]):
        y_true = targets_np[:, i]
        y_score = predictions_np[:, i]
        
        if len(np.unique(y_true)) <= 1:
            optimal_thresholds[i] = 0.5
            continue
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        
        if method == 'f1':
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores)
            optimal_thresholds[i] = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        elif method == 'precision':
            # 找到召回率>=0.8的最高精确率阈值
            valid_idx = recall >= 0.8
            if valid_idx.any():
                best_precision = precision[valid_idx].max()
                optimal_thresholds[i] = float(thresholds[valid_idx][precision[valid_idx] == best_precision][0])
            else:
                optimal_thresholds[i] = 0.5
        elif method == 'recall':
            # 找到精确率>=0.8的最高召回率阈值
            valid_idx = precision >= 0.8
            if valid_idx.any():
                best_recall = recall[valid_idx].max()
                optimal_thresholds[i] = float(thresholds[valid_idx][recall[valid_idx] == best_recall][0])
            else:
                optimal_thresholds[i] = 0.5
        else:
            optimal_thresholds[i] = 0.5
    
    return optimal_thresholds


def compute_average_precision(predictions: torch.Tensor,
                             targets: torch.Tensor) -> float:
    """
    计算平均精确率 (AP)
    
    Args:
        predictions: 预测概率
        targets: 真实标签
    
    Returns:
        AP值
    """
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    try:
        return average_precision_score(targets_np, predictions_np)
    except:
        return 0.0

