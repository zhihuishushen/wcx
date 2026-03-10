"""
双层图模型包

提供继承关系增强的漏洞检测模型：
- DualLayerGNNModel: 完整的双层图模型
- DualLayerGraphClassifier: 简化版双层图分类器
- MultiTaskDualLayerModel: 多任务双层图模型
- create_model: 模型工厂函数
"""

from .dual_layer_gnn import (
    DualLayerGNNModel,
    UpperGraphEncoder,
    DualLayerGraphClassifier,
    MultiTaskDualLayerModel,
    create_model,
)

__all__ = [
    'DualLayerGNNModel',
    'UpperGraphEncoder',
    'DualLayerGraphClassifier',
    'MultiTaskDualLayerModel',
    'create_model',
]











