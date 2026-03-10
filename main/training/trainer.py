"""
训练器模块
实现两阶段训练流程
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from tqdm import tqdm
import logging

from .loss import HybridLoss, MultiTaskLoss
from .evaluator import Evaluator


class Trainer:
    """
    训练器
    
    功能：
    1. 两阶段训练（预训练 + 微调）
    2. 早停和学习率调度
    3. 模型保存和加载
    4. 日志记录
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 test_dataset: Dataset,
                 config: Dict,
                 device: str = 'cuda',
                 loss_fn: nn.Module = None,
                 vulnerability_types: List[str] = None):
        """
        Args:
            model: 模型
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            test_dataset: 测试数据集
            config: 配置字典
            device: 设备
            loss_fn: 自定义损失函数（可选）
            vulnerability_types: 漏洞类型列表（可选）
        """
        self.device = device
        self.model = model.to(device)
        
        self.config = config
        self.vulnerability_types = vulnerability_types or config.get('vulnerability_types', 
            ['reentrancy', 'unchecked external call', 'ether frozen', 'ether strict equality'])
        self.num_classes = len(self.vulnerability_types)
        
        # 数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 16),
            shuffle=True,
            num_workers=config.get('num_workers', 0),
            collate_fn=self._collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 16),
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            collate_fn=self._collate_fn
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.get('batch_size', 16),
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            collate_fn=self._collate_fn
        )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=config.get('lr_patience', 5),
            verbose=False
        )
        
        # 损失函数
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = HybridLoss(
                baseline_weight=config.get('baseline_weight', 1.0),
                pseudo_weight=config.get('pseudo_weight', 0.5),
                confidence_weight=config.get('confidence_weight', 0.1)
            )
        
        # 评估器
        self.evaluator = Evaluator(vulnerability_types=self.vulnerability_types)
        
        # 输出目录
        self.output_dir = Path(config.get('output_dir', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志
        self.logger = self._setup_logger()
        
        # 最佳指标
        self.best_metric = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_metrics': [],
            'learning_rates': []
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('Trainer')
        logger.setLevel(logging.INFO)
        
        # 文件处理器
        log_file = self.output_dir / 'training.log'
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """
        自定义collate函数
        将样本列表转换为模型输入格式
        """
        if not batch:
            return {}
        
        # 处理单个样本（简化版）
        processed_batch = []
        
        for sample in batch:
            # 解析样本数据
            target_graph = sample.get('target_graph', {})
            upper_graphs = sample.get('upper_graphs', [])
            labels = sample.get('labels', torch.zeros(self.num_classes))
            
            # 构建下层图输入
            node_features = target_graph.get('node_features', torch.tensor([]))
            edge_index = target_graph.get('edge_index', torch.tensor([[], []]))
            edge_type = target_graph.get('edge_type', torch.tensor([]))
            
            # 如果是numpy数组，转换为tensor
            if not isinstance(node_features, torch.Tensor):
                node_features = torch.tensor(node_features, dtype=torch.float32)
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, dtype=torch.long)
            if not isinstance(edge_type, torch.Tensor):
                edge_type = torch.tensor(edge_type, dtype=torch.long)
            
            # 构建下层图
            lower_graph = {
                'node_features': node_features,
                'edge_index': edge_index,
                'edge_type': edge_type,
                'graph_nodes_list': None,
                'num_graphs': 1
            }
            
            # 处理上层图
            has_upper = len(upper_graphs) > 0 and upper_graphs[0].get('num_nodes', 0) > 0
            upper_graph = None
            
            if has_upper:
                ug = upper_graphs[0]
                ug_features = ug.get('node_features')
                ug_edges = ug.get('edge_index', [])
                
                if not isinstance(ug_features, torch.Tensor):
                    ug_features = torch.tensor(ug_features, dtype=torch.float32)
                if ug_edges is not None and len(ug_edges) > 0:
                    if not isinstance(ug_edges, torch.Tensor):
                        ug_edges = torch.tensor(ug_edges, dtype=torch.long)
                else:
                    ug_edges = torch.tensor([[], []], dtype=torch.long)
                
                upper_graph = {
                    'node_features': ug_features,
                    'edge_index': ug_edges
                }
            
            # 处理标签
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.float32)
            
            # 移到设备上
            lower_graph = {
                'node_features': lower_graph['node_features'].to(self.device),
                'edge_index': lower_graph['edge_index'].to(self.device),
                'edge_type': lower_graph['edge_type'].to(self.device),
                'graph_nodes_list': lower_graph.get('graph_nodes_list'),
                'num_graphs': lower_graph.get('num_graphs', 1)
            }
            
            if upper_graph is not None:
                upper_graph = {
                    'node_features': upper_graph['node_features'].to(self.device),
                    'edge_index': upper_graph['edge_index'].to(self.device)
                }
            
            labels = labels.to(self.device)
            
            processed_batch.append({
                'lower_graph': lower_graph,
                'upper_graph': upper_graph,
                'has_upper': has_upper,
                'labels': labels
            })
        
        return processed_batch
    
    def train(self, epochs: int = None) -> Dict:
        """
        训练模型
        
        Args:
            epochs: 训练轮数
        
        Returns:
            训练历史
        """
        epochs = epochs or self.config.get('epochs', 50)
        pretrain_epochs = self.config.get('pretrain_epochs', 5)
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Pretrain epochs: {pretrain_epochs}")
        self.logger.info(f"Batch size: {self.config.get('batch_size', 16)}")
        self.logger.info(f"Learning rate: {self.config.get('learning_rate', 1e-4)}")
        self.logger.info(f"Vulnerability types: {self.vulnerability_types}")
        
        for epoch in range(epochs):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            self.logger.info(f"{'='*50}")
            
            # 判断是否在预训练阶段
            is_pretrain = epoch < pretrain_epochs
            phase = "Pretrain" if is_pretrain else "Finetune"
            self.logger.info(f"Phase: {phase}")
            
            # 训练
            train_loss = self._train_epoch(epoch, is_pretrain)
            self.history['train_loss'].append(train_loss)
            
            # 验证
            val_metrics = self._validate(epoch, is_pretrain)
            self.history['val_metrics'].append(val_metrics)
            
            # 学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            self.scheduler.step(val_metrics.get('macro_f1', 0))
            
            # 早停检查
            if val_metrics.get('macro_f1', 0) > self.best_metric:
                self.best_metric = val_metrics.get('macro_f1', 0)
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch, 'best')
                self.logger.info(f"New best model saved! F1: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config.get('patience', 10):
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # 保存定期检查点
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self._save_checkpoint(epoch, f'epoch_{epoch + 1}')
        
        # 最终测试评估
        self.logger.info("\n" + "="*50)
        self.logger.info("Final Evaluation on Test Set")
        self.logger.info("="*50)
        test_results = self.evaluate(self.test_loader)
        self._save_results(test_results, 'final_test_results.json')
        
        # 保存训练历史
        self._save_history()
        
        return self.history
    
    def _train_epoch(self, epoch: int, is_pretrain: bool) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_samples in progress_bar:
            # 处理批次中的每个样本
            batch_predictions = []
            batch_targets = []
            
            self.optimizer.zero_grad()
            
            for sample in batch_samples:
                # 获取输入
                lower_graph = sample['lower_graph']
                upper_graph = sample['upper_graph']
                has_upper = sample['has_upper']
                labels = sample['labels'].to(self.device)
                
                # 前向传播
                try:
                    results = self.model(lower_graph, upper_graph, has_upper)
                    predictions = results['predictions']  # [1, num_classes]
                    
                    # 应用sigmoid得到概率
                    predictions_prob = torch.sigmoid(predictions)
                    
                    batch_predictions.append(predictions_prob)
                    batch_targets.append(labels.unsqueeze(0))
                    
                except Exception as e:
                    # 跳过处理失败的样本
                    self.logger.warning(f"Forward pass failed: {e}")
                    continue
            
            if len(batch_predictions) == 0:
                continue
            
            # 合并批次
            predictions = torch.cat(batch_predictions, dim=0)  # [batch, num_classes]
            targets = torch.cat(batch_targets, dim=0)  # [batch, num_classes]
            
            # 计算损失
            loss_dict = self.loss_fn(predictions, targets)
            loss = loss_dict['total_loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}'
            })
        
        return total_loss / max(num_batches, 1)
    
    def _validate(self, epoch: int, is_pretrain: bool) -> Dict:
        """验证模型"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_samples in tqdm(self.val_loader, desc="Validating"):
                for sample in batch_samples:
                    lower_graph = sample['lower_graph']
                    upper_graph = sample['upper_graph']
                    has_upper = sample['has_upper']
                    labels = sample['labels'].to(self.device)
                    
                    try:
                        results = self.model(lower_graph, upper_graph, has_upper)
                        predictions = torch.sigmoid(results['predictions'])
                        
                        all_predictions.append(predictions)
                        all_targets.append(labels.unsqueeze(0))
                    except Exception as e:
                        self.logger.warning(f"Validation forward failed: {e}")
                        continue
        
        # 计算评估指标
        if all_predictions and len(all_predictions) > 0:
            predictions = torch.cat(all_predictions, dim=0)
            targets = torch.cat(all_targets, dim=0)
            
            metrics = self.evaluator.evaluate(predictions, targets)
            
            self.logger.info(f"\nValidation Results (Epoch {epoch + 1}):")
            self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            self.logger.info(f"  Macro F1: {metrics['macro']['f1']:.4f}")
            self.logger.info(f"  Macro Precision: {metrics['macro']['precision']:.4f}")
            self.logger.info(f"  Macro Recall: {metrics['macro']['recall']:.4f}")
            
            return {
                'macro_f1': metrics['macro']['f1'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['macro']['precision'],
                'recall': metrics['macro']['recall']
            }
        
        return {'macro_f1': 0.0, 'accuracy': 0.0}
    
    def evaluate(self, dataloader: DataLoader = None) -> Dict:
        """评估模型"""
        if dataloader is None:
            dataloader = self.test_loader
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_samples in tqdm(dataloader, desc="Evaluating"):
                for sample in batch_samples:
                    lower_graph = sample['lower_graph']
                    upper_graph = sample['upper_graph']
                    has_upper = sample['has_upper']
                    labels = sample['labels'].to(self.device)
                    
                    try:
                        results = self.model(lower_graph, upper_graph, has_upper)
                        predictions = torch.sigmoid(results['predictions'])
                        
                        all_predictions.append(predictions)
                        all_targets.append(labels.unsqueeze(0))
                    except Exception as e:
                        self.logger.warning(f"Evaluation forward failed: {e}")
                        continue
        
        if all_predictions and len(all_predictions) > 0:
            predictions = torch.cat(all_predictions, dim=0)
            targets = torch.cat(all_targets, dim=0)
            
            metrics = self.evaluator.evaluate(predictions, targets)
            
            self.logger.info(f"\nTest Results:")
            self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            self.logger.info(f"  Macro F1: {metrics['macro']['f1']:.4f}")
            
            return metrics
        
        return {}
    
    def _save_checkpoint(self, epoch: int, suffix: str):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
            'history': self.history
        }
        
        checkpoint_path = self.output_dir / f'model_{suffix}.pth'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_results(self, results: Dict, filename: str):
        """保存评估结果"""
        results_path = self.output_dir / filename
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Results saved: {results_path}")
    
    def _save_history(self):
        """保存训练历史"""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        self.logger.info(f"History saved: {history_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_metric = checkpoint.get('best_metric', 0.0)
        self.best_epoch = checkpoint.get('epoch', 0)
        self.history = checkpoint.get('history', {})
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.best_epoch + 1}")
        
        return self.best_epoch


class StagedTrainer(Trainer):
    """
    分阶段训练器
    
    特点：
    1. 阶段A：仅使用基础图训练
    2. 阶段B：选择性使用上层图微调
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.pretrain_epochs = self.config.get('pretrain_epochs', 5)
        self.finetune_epochs = self.config.get('finetune_epochs', 45)
    
    def train(self) -> Dict:
        """分阶段训练"""
        total_epochs = self.pretrain_epochs + self.finetune_epochs
        
        self.logger.info("=" * 60)
        self.logger.info("Stage A: Pretraining with lower-only graph")
        self.logger.info("=" * 60)
        
        # 阶段A：仅使用基础图
        pretrain_history = super().train(epochs=self.pretrain_epochs)
        
        # 加载最佳模型
        self.load_checkpoint(str(self.output_dir / 'model_best.pth'))
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Stage B: Fine-tuning with selective upper graph")
        self.logger.info("=" * 60)
        
        # 阶段B：选择性使用上层图
        finetune_history = super().train(epochs=total_epochs)
        
        # 合并历史
        self.history = {
            'pretrain': pretrain_history,
            'finetune': finetune_history
        }
        
        return self.history
