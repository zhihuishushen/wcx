"""
项目配置文件
包含模型、训练、门控等所有配置
"""

from pathlib import Path
import os

# 基础路径配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "samples"
OUTPUT_DIR = BASE_DIR / "output"

# 模型配置
MODEL_CONFIG = {
    'hidden_dim': 128,
    'num_classes': 4,  # 4种漏洞类型
    'node_feature_dim': 64,  # 节点特征维度
    'edge_feature_dim': 32,  # 边特征维度
    'num_gnn_layers': 3,  # GNN层数
    'num_heads': 4,  # 注意力头数
    'dropout': 0.3,
    'use_upper_graph': True,  # 是否使用上层图
    'gate_type': 'adaptive',  # 门控类型: 'adaptive', 'simple', 'always', 'never'
    'upper_graph_dim': 64,  # 上层图特征维度
}

# 门控模块配置
GATE_CONFIG = {
    'use_learnable': True,  # 是否使用可学习门控
    'gate_type': 'adaptive',  # 'fixed', 'learnable', 'adaptive'
    'gate_threshold': 0.5,  # 固定门控阈值
    'gate_hidden_dim': 64,
    'temperature': 1.0,  # 温度参数
}

# 上层图配置
UPPER_GRAPH_CONFIG = {
    'suspicious_score_threshold': 0.3,
    'max_upper_nodes': 100,
    'top_k_dependencies': 5,
    'connection_strength_threshold': 0.1,
    'always_build_upper': True,  # 是否始终构建上层图（让门控决定是否使用）
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 16,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'pretrain_epochs': 5,  # 预训练轮数
    'finetune_epochs': 45,  # 微调轮数
    'patience': 10,  # 早停耐心值
    'lr_patience': 5,  # 学习率调度耐心值
    'lr_factor': 0.5,  # 学习率衰减因子
    'num_workers': 0,
    'save_interval': 10,
}

# 损失函数配置
LOSS_CONFIG = {
    'baseline_weight': 1.0,  # 基础损失权重
    'pseudo_weight': 0.5,  # 伪标签损失权重
    'confidence_weight': 0.1,  # 置信度损失权重
    'gate_weight': 0.1,  # 门控损失权重
    'use_focal': True,  # 是否使用Focal Loss
    'focal_gamma': 2.0,  # Focal Loss gamma参数
    'gate_regularization': 0.01,  # 门控正则化强度
}

# 可疑分数配置
SUSPICIOUS_SCORE_CONFIG = {
    'version': 'v3',  # 'v1', 'v2', 'v3'
    'per_type_thresholds': {
        'reentrancy': 0.5,
        'unchecked external call': 0.4,
        'ether frozen': 0.3,
        'ether strict equality': 0.3,
    }
}

# 漏洞类型配置
VULNERABILITY_CONFIG = {
    'types': [
        'reentrancy',
        'unchecked external call',
        'ether frozen',
        'ether strict equality'
    ],
    'classes': {
        'A_class': [
            'reentrancy',
            'unchecked external call',
            'ether frozen',
            'ether strict equality'
        ],
        'B_class': [],  # 暂时为空
        'C_class': [],  # 暂时为空
    }
}

# 继承关系配置
INHERITANCE_CONFIG = {
    'decay_params': {
        'reentrancy': {'decay': 0.4, 'severity': 1.0},
        'unchecked external call': {'decay': 0.3, 'severity': 0.9},
        'ether frozen': {'decay': 0.6, 'severity': 0.7},
        'ether strict equality': {'decay': 0.8, 'severity': 0.5},
    },
    'connection_weights': {
        'inherit': 1.0,
        'call': 0.7,
        'import': 0.5,
    }
}

# 数据配置
DATA_CONFIG = {
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'stratify': True,  # 是否分层采样
}

# 评估配置
EVAL_CONFIG = {
    'default_threshold': 0.5,
    'threshold_method': 'f1',  # 'f1', 'precision', 'recall'
}

# 输出配置
OUTPUT_CONFIG = {
    'save_model': True,
    'save_results': True,
    'save_logs': True,
    'log_level': 'INFO',
}

