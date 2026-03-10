"""
Baseline GNN模块

提供原版GNNSCVulDetector的PyTorch实现：
- GGNNMessagePassing: GGNN消息传递层
- BasicGNNEncoder: 基础GNN编码器
- GGNNEncoder: 完整GGNN编码器
- GatedRegression: 门控回归层
- BaselineGGNNEncoder: 完整基线模型
- SimpleGraphEncoder: 简化图编码器（测试用）
"""

from .gnn_encoder import (
    GGNNMessagePassing,
    BasicGNNEncoder,
    GGNNEncoder,
    GatedRegression,
    BaselineGGNNEncoder,
    SimpleGraphEncoder,
)

__all__ = [
    'GGNNMessagePassing',
    'BasicGNNEncoder',
    'GGNNEncoder',
    'GatedRegression',
    'BaselineGGNNEncoder',
    'SimpleGraphEncoder',
]











