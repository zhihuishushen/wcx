"""
继承关系模块
用于提取和分析智能合约之间的继承关系
"""

from .extractor import InheritanceExtractor
from .risk_scorer import EnhancedRiskScorer
from .analyzer import InheritanceAnalyzer

__all__ = ['InheritanceExtractor', 'EnhancedRiskScorer', 'InheritanceAnalyzer']

