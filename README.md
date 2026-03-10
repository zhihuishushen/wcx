# New-Project: 继承关系增强的智能合约漏洞检测

## 概述

本项目实现了一个保守的继承关系增强检测方案，核心特点是：
1. **保留原版GNNSCVulDetector精度**：基础检测路径始终使用原版图编码
2. **可学习门控**：智能判断何时使用继承关系信息
3. **条件增强**：仅在高置信度时构建和使用上层图

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                      输入层                              │
│  ┌─────────────┐  ┌─────────────────────────────────┐  │
│  │ 合约源代码  │→│  图提取器 + 继承关系提取器      │  │
│  └─────────────┘  └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                   原版检测路径（始终启用）                │
│  ┌─────────────────────────────────────────────────┐  │
│  │           BaselineGGNNEncoder                   │  │
│  │           (复用GNNSCVulDetector)                │  │
│  └─────────────────────────────────────────────────┘  │
│                          ↓                              │
│              BaselineClassifier → 基础预测                │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                   增强检测路径（条件启用）                │
│  ┌─────────────────────────────────────────────────┐  │
│  │           InheritanceAnalyzer                   │  │
│  │           (继承关系分析)                        │  │
│  └─────────────────────────────────────────────────┘  │
│                          ↓                              │
│  ┌─────────────────────────────────────────────────┐  │
│  │           LearnableGate (可学习门控)            │  │
│  │           输出：gate信号 + 置信度                │  │
│  └─────────────────────────────────────────────────┘  │
│                          ↓                              │
│           ┌─────────────────┬─────────────────┐        │
│           ↓                 ↓                 ↓        │
│      置信度高           置信度低          门控输出   │
│           ↓                 ↓                 ↓        │
│    UpperGraphBuilder    Skip Upper        预测融合    │
│    (构建上层图)          Graph             Confidence  │
│           ↓                 ↓                 ↓        │
│    DualGraphFusion ──→ PredictionFusion               │
│                          ↓                             │
│                   最终预测结果                          │
└─────────────────────────────────────────────────────────┘
```

## 模块说明

### 1. 继承关系模块 (`main/inheritance/`)

| 文件 | 功能 |
|-----|------|
| `extractor.py` | 从Solidity代码提取继承关系 |
| `risk_scorer.py` | 可疑分数计算器（支持V1/V2/V3公式） |
| `analyzer.py` | 继承关系分析器，整合提取和风险评估 |

### 2. 门控模块 (`main/gating/`)

| 文件 | 功能 |
|-----|------|
| `learnable_gate.py` | 可学习门控（LearnableGate, AdaptiveGate, FixedGate） |
| `confidence.py` | 置信度计算器 |

### 3. 上层图模块 (`main/upper_graph/`)

| 文件 | 功能 |
|-----|------|
| `builder.py` | 上层图构建器 |
| `fusion.py` | 双层图融合（PredictionFusion, DualGraphFusion, HierarchicalFusion） |

### 4. 训练模块 (`main/training/`)

| 文件 | 功能 |
|-----|------|
| `trainer.py` | 训练器（支持两阶段训练） |
| `loss.py` | 损失函数（FocalLoss, HybridLoss, GateLoss） |
| `evaluator.py` | 评估器 |

### 5. 工具模块 (`utils/`)

| 文件 | 功能 |
|-----|------|
| `data_loader.py` | 数据加载器 |
| `metrics.py` | 评估指标计算 |

## 配置说明

### 核心配置 (`optimized_config.py`)

```python
# 模型配置
MODEL_CONFIG = {
    'hidden_dim': 128,
    'num_classes': 4,
    'use_upper_graph': True,
}

# 门控配置
GATE_CONFIG = {
    'use_learnable': True,
    'gate_type': 'adaptive',  # fixed, learnable, adaptive
    'gate_threshold': 0.5,
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 16,
    'num_epochs': 50,
    'pretrain_epochs': 5,  # 预训练阶段
    'finetune_epochs': 45,  # 微调阶段
}
```

## 使用方法

### 1. 数据准备

将JSON格式的样本数据放入 `data/samples/` 目录。

样本格式：
```json
{
    "target_contract": "ContractName",
    "target_graph": {
        "node_features": [[...]],
        "edge_index": [[...], [...]],
        "edge_type": [...]
    },
    "upper_graphs": [...],
    "cross_edges": [...],
    "labels": {
        "reentrancy": 0,
        "unchecked external call": 1,
        "ether frozen": 0,
        "ether strict equality": 0
    },
    "pseudo_labels": {...},
    "has_upper_graph": true,
    "suspicious_score": 0.65
}
```

### 2. 运行训练

```bash
cd new-project
python run_training.py --samples data/samples
```

### 3. 运行评估

```python
from main.training.evaluator import Evaluator

evaluator = Evaluator()
results = evaluator.evaluate(predictions, targets)
report = evaluator.generate_report(results)
```

## 可疑分数计算（V3版本）

```
score = base_score × severity × weight × path_bonus × complexity × depth_factor

其中：
- base_score = exp(-decay × distance)  # 指数衰减
- path_bonus = 1 + 0.2 × log(1 + path_count)  # 对数增长
- complexity = 1 + 0.3 × degree + 0.2 × connectivity  # 复杂度因子
- depth_factor = 1 + 0.1 × inheritance_depth  # 继承深度因子
```

## 文件结构

```
new-project/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包
├── optimized_config.py           # 配置文件
├── run_training.py             # 运行脚本
├── data/                       # 数据目录
│   └── samples/                # 样本数据
├── main/                       # 核心模块
│   ├── __init__.py
│   ├── config.py              # 运行时配置
│   ├── inheritance/           # 继承关系模块
│   │   ├── __init__.py
│   │   ├── extractor.py
│   │   ├── risk_scorer.py
│   │   └── analyzer.py
│   ├── gating/               # 门控模块
│   │   ├── __init__.py
│   │   ├── learnable_gate.py
│   │   └── confidence.py
│   ├── upper_graph/          # 上层图模块
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   └── fusion.py
│   ├── training/             # 训练模块
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── loss.py
│   │   └── evaluator.py
│   └── experimental/         # 实验工具
│       ├── ablation_runner.py
│       └── data_stats.py
├── utils/                    # 工具模块
│   ├── __init__.py
│   ├── data_loader.py
│   └── metrics.py
└── output/                   # 输出目录
    ├── models/              # 模型保存
    ├── logs/               # 训练日志
    └── results/            # 评估结果
```

## 依赖包

```
torch>=1.9.0
numpy>=1.19.0
networkx>=2.5
scikit-learn>=0.24.0
pandas>=1.2.0
tqdm>=4.60.0
```

## 实验建议

1. **Baseline实验**：使用固定门控（gate_type='fixed'）验证基础性能
2. **可学习门控实验**：使用adaptive类型，自动学习何时使用上层图
3. **消融实验**：
   - 移除上层图（use_upper_graph=False）
   - 移除门控（use_learnable=False）
   - 不同可疑分数公式对比（version='v1'/'v2'/'v3'）

## 注意事项

1. 本项目为保守增强方案，确保在继承关系信息不可靠时退化为原版检测
2. 门控模块会自动学习是否使用上层图，无需手动设置阈值
3. 建议先在小数据集上验证流程，再使用完整数据训练
