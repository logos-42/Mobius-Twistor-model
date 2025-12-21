"""
MT-Transformer: Möbius-Twistor Transformer Components
集成到HOPE架构的核心组件，以及HOPE架构迁移到扭量莫比乌斯架构
"""

from .spinor_embedding import SpinorEmbedding
from .incidence_attention import IncidenceAttention
from .mobius_layer import MobiusLayer
from .nested_optimizer import NestedOptimizer
from .hope_integration import HopeIntegration

# HOPE架构迁移到扭量莫比乌斯架构的新组件
from .twistor_self_modifying import (
    TwistorSelfModifying,
    TwistorEvolution,
    AdaptiveIncidenceAttention,
    AdaptiveMobiusLayer
)
from .twistor_memory_system import (
    TwistorMemorySystem,
    TwistorMemoryState,
    MobiusMemoryCycle,
    PhaseCompression
)
from .twistor_nested_learning import (
    TwistorNestedLearning,
    OmegaNestedOptimizer,
    PiNestedOptimizer,
    IncidenceConstraint
)
from .mt_hope_architecture import MTHopeArchitecture

# 扭量化HOPE架构（Recurrent版本，无注意力）
from .twistor_titans_cell import TwistorTitansCell
from .twistor_titans_recurrent import TwistorTitansRecurrent
from .twistor_self_modifying_recurrent import TwistorSelfModifyingRecurrent
from .twistor_hope_architecture import TwistorHopeArchitecture

__all__ = [
    # 基础组件
    'SpinorEmbedding',
    'IncidenceAttention',
    'MobiusLayer',
    'NestedOptimizer',
    'HopeIntegration',
    # 扭量自我修正
    'TwistorSelfModifying',
    'TwistorEvolution',
    'AdaptiveIncidenceAttention',
    'AdaptiveMobiusLayer',
    # 扭量记忆系统
    'TwistorMemorySystem',
    'TwistorMemoryState',
    'MobiusMemoryCycle',
    'PhaseCompression',
    # 扭量嵌套学习
    'TwistorNestedLearning',
    'OmegaNestedOptimizer',
    'PiNestedOptimizer',
    'IncidenceConstraint',
    # 完整架构
    'MTHopeArchitecture',
    # 扭量化HOPE架构（Recurrent版本）
    'TwistorTitansCell',
    'TwistorTitansRecurrent',
    'TwistorSelfModifyingRecurrent',
    'TwistorHopeArchitecture',
]

__version__ = '0.1.4'

