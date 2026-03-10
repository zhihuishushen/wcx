"""
评估器模块
计算多标签分类评估指标
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, precision_recall_curve,
    confusion_matrix, classification_report
)
import json


class Evaluator:
    """
    评估器
    
    功能：
    1. 计算多标签分类指标
    2. 生成评估报告
    3. 支持门控评估
    """
    
    def __init__(self, vulnerability_types: List[str] = None):
        """
        Args:
            vulnerability_types: 漏洞类型列表
        """
        self.vulnerability_types = vulnerability_types or [
            'reentrancy',
            'unchecked external call',
            'ether frozen',
            'ether strict equality'
        ]
    
    def evaluate(self, predictions: torch.Tensor, 
                 targets: torch.Tensor,
                 thresholds: Optional[Dict[str, float]] = None) -> Dict:
        """
        评估模型性能
        
        Args:
            predictions: 预测概率 [batch, num_classes]
            targets: 真实标签 [batch, num_classes]
            thresholds: 每个类型的阈值
        
        Returns:
            评估结果字典
        """
        # 检查输入是否有效
        if predictions.numel() == 0 or targets.numel() == 0:
            return {
                'accuracy': 0.0,
                'macro': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0},
                'weighted': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                'per_type': {},
                'sample_level': {'exact_match': 0.0, 'partial_match': 0.0}
            }
        
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        # 应用阈值获取二值预测
        if thresholds is None:
            thresholds = {vuln_type: 0.5 for vuln_type in self.vulnerability_types}
        
        binary_preds = np.zeros_like(predictions_np)
        for i, vuln_type in enumerate(self.vulnerability_types):
            threshold = thresholds.get(vuln_type, 0.5)
            binary_preds[:, i] = (predictions_np[:, i] >= threshold).astype(float)
        
        results = {}
        
        # 1. 整体指标
        results['accuracy'] = accuracy_score(targets_np.flatten(), binary_preds.flatten())
        
        # 2. 每类指标
        per_type_metrics = {}
        for i, vuln_type in enumerate(self.vulnerability_types):
            y_true = targets_np[:, i]
            y_pred = binary_preds[:, i]
            y_score = predictions_np[:, i]
            
            # 计算指标
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            # AUC
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_score)
            else:
                auc = 0.5  # 无法计算AUC时
            
            per_type_metrics[vuln_type] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'auc': float(auc),
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            }
        
        results['per_type'] = per_type_metrics
        
        # 3. 宏观平均
        if per_type_metrics:
            results['macro'] = {
                'precision': np.mean([m['precision'] for m in per_type_metrics.values()]),
                'recall': np.mean([m['recall'] for m in per_type_metrics.values()]),
                'f1': np.mean([m['f1'] for m in per_type_metrics.values()]),
                'auc': np.mean([m['auc'] for m in per_type_metrics.values()])
            }
        else:
            results['macro'] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc': 0.0
            }
        
        # 4. 加权平均
        pos_counts = targets_np.sum(axis=0)
        weights = pos_counts / (pos_counts.sum() + 1e-8)
        if pos_counts.sum() > 0 and per_type_metrics:
            results['weighted'] = {
                'precision': np.average([m['precision'] for m in per_type_metrics.values()], weights=weights),
                'recall': np.average([m['recall'] for m in per_type_metrics.values()], weights=weights),
                'f1': np.average([m['f1'] for m in per_type_metrics.values()], weights=weights)
            }
        else:
            results['weighted'] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        # 5. 样本级指标
        exact_match = (binary_preds == targets_np).all(axis=1).mean()
        partial_match = ((binary_preds == targets_np).sum(axis=1) / targets_np.shape[1]).mean()
        
        results['sample_level'] = {
            'exact_match': float(exact_match),
            'partial_match': float(partial_match)
        }
        
        return results
    
    def evaluate_with_gate(self, 
                          predictions: torch.Tensor,
                          targets: torch.Tensor,
                          use_upper: torch.Tensor,
                          thresholds: Optional[Dict[str, float]] = None) -> Dict:
        """
        评估带门控的模型
        
        Args:
            predictions: 预测概率
            targets: 真实标签
            use_upper: 门控信号
            thresholds: 预测阈值
        
        Returns:
            评估结果
        """
        # 基础评估
        results = self.evaluate(predictions, targets, thresholds)
        
        # 门控评估
        use_upper_np = use_upper.detach().cpu().numpy()
        
        results['gate'] = {
            'mean': float(use_upper_np.mean()),
            'std': float(use_upper_np.std()),
            'min': float(use_upper_np.min()),
            'max': float(use_upper_np.max()),
            'upper_rate': float((use_upper_np > 0.5).mean())
        }
        
        return results
    
    def find_optimal_thresholds(self, 
                               predictions: torch.Tensor,
                               targets: torch.Tensor,
                               method: str = 'f1') -> Dict[str, float]:
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
        
        for i, vuln_type in enumerate(self.vulnerability_types):
            y_true = targets_np[:, i]
            y_score = predictions_np[:, i]
            
            if len(np.unique(y_true)) <= 1:
                # 只有一个类别，使用默认阈值
                optimal_thresholds[vuln_type] = 0.5
                continue
            
            # 计算PR曲线
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            
            if method == 'f1':
                # F1最大点
                f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
                best_idx = np.argmax(f1_scores)
                optimal_thresholds[vuln_type] = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
            
            elif method == 'precision':
                # 召回率约束下的精确率最大
                optimal_thresholds[vuln_type] = self._find_threshold_with_recall(
                    y_true, y_score, min_recall=0.8
                )
            
            elif method == 'recall':
                # 精确率约束下的召回率最大
                optimal_thresholds[vuln_type] = self._find_threshold_with_precision(
                    y_true, y_score, min_precision=0.8
                )
            
            else:
                optimal_thresholds[vuln_type] = 0.5
        
        return optimal_thresholds
    
    def _find_threshold_with_recall(self, y_true: np.ndarray, 
                                    y_score: np.ndarray,
                                    min_recall: float = 0.8) -> float:
        """找到满足最小召回率的阈值"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        
        valid_idx = recall >= min_recall
        if not valid_idx.any():
            return 0.5
        
        best_precision = precision[valid_idx].max()
        return float(thresholds[valid_idx][precision[valid_idx] == best_precision][0])
    
    def _find_threshold_with_precision(self, y_true: np.ndarray,
                                       y_score: np.ndarray,
                                       min_precision: float = 0.8) -> float:
        """找到满足最小精确率的阈值"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        
        valid_idx = precision >= min_precision
        if not valid_idx.any():
            return 0.5
        
        best_recall = recall[valid_idx].max()
        return float(thresholds[valid_idx][recall[valid_idx] == best_recall][0])
    
    def generate_report(self, results: Dict, output_path: str = None) -> str:
        """
        生成文本报告
        
        Args:
            results: 评估结果
            output_path: 输出文件路径
        
        Returns:
            报告字符串
        """
        lines = []
        
        lines.append("=" * 60)
        lines.append("智能合约漏洞检测评估报告")
        lines.append("=" * 60)
        lines.append("")
        
        # 整体指标
        lines.append("【整体指标】")
        lines.append(f"  准确率 (Accuracy): {results['accuracy']:.4f}")
        lines.append("")
        
        # 宏观平均
        lines.append("【宏观平均 (Macro Average)】")
        lines.append(f"  精确率 (Precision): {results['macro']['precision']:.4f}")
        lines.append(f"  召回率 (Recall): {results['macro']['recall']:.4f}")
        lines.append(f"  F1分数: {results['macro']['f1']:.4f}")
        lines.append(f"  AUC: {results['macro']['auc']:.4f}")
        lines.append("")
        
        # 每类指标
        lines.append("【每类指标】")
        for vuln_type, metrics in results['per_type'].items():
            lines.append(f"  {vuln_type}:")
            lines.append(f"    Precision: {metrics['precision']:.4f}")
            lines.append(f"    Recall: {metrics['recall']:.4f}")
            lines.append(f"    F1: {metrics['f1']:.4f}")
            lines.append(f"    AUC: {metrics['auc']:.4f}")
            lines.append(f"    TP/FP/TN/FN: {metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']}")
        lines.append("")
        
        # 样本级指标
        lines.append("【样本级指标】")
        lines.append(f"  完全匹配率: {results['sample_level']['exact_match']:.4f}")
        lines.append(f"  部分匹配率: {results['sample_level']['partial_match']:.4f}")
        lines.append("")
        
        # 门控信息（如果有）
        if 'gate' in results:
            lines.append("【门控统计】")
            lines.append(f"  平均门控值: {results['gate']['mean']:.4f}")
            lines.append(f"  上层图使用率: {results['gate']['upper_rate']:.4f}")
            lines.append("")
        
        lines.append("=" * 60)
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    def save_results(self, results: Dict, output_path: str):
        """
        保存评估结果到JSON文件
        
        Args:
            results: 评估结果
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

