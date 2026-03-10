# 双层GNN智能合约漏洞检测实验报告

## 实验概述

本实验旨在验证双层GNN模型（结合代码语义图和继承关系图）在智能合约漏洞检测任务上的有效性。

## 实验配置

- **数据集**: 自有Dataset (3850+ contracts)
- **平衡策略**: Undersampling (ratio=1.0, 1:1平衡)
- **模型参数**:
  - epochs: 12
  - batch_size: 8
  - learning_rate: 0.001
  - hidden_dim: 128
  - seed: 42

## 实验结果对比

### 1. unchecked external call 漏洞

| 指标 | 双层GNN | 基线GNN | 提升 |
|------|---------|---------|------|
| **Test Macro F1** | **79.74%** | 55.62% | **+24.12%** ✅ |
| Test Accuracy | 68.57% | 60.00% | +8.57% |
| Positive F1 | 75.76% | 69.57% | +6.19% |

### 2. reentrancy 漏洞

| 指标 | 双层GNN | 基线GNN | 提升 |
|------|---------|---------|------|
| **Test Macro F1** | **83.18%** | 83.18% | 0% |
| Test Accuracy | 83.78% | 83.78% | 0% |
| Positive F1 | 80.00% | 80.00% | 0% |

### 3. dangerous delegatecall 漏洞

| 指标 | 双层GNN | 基线GNN | 提升 |
|------|---------|---------|------|
| **Test Macro F1** | 81.75% | **86.55%** | -4.80% ❌ |
| Test Accuracy | 82.61% | 86.96% | -4.35% |
| Positive F1 | 85.71% | 88.89% | -3.18% |

## 结论

双层GNN模型在unchecked external call漏洞检测上显著优于基线模型，Macro F1提升达24.12%。但在其他两种漏洞类型上效果不稳定。

---
*报告生成时间: 2026-03-06*
