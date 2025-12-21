"""
MT-Transformer: Möbius-Twistor Transformer Components
集成到HOPE架构的核心组件
"""

from .spinor_embedding import SpinorEmbedding
from .incidence_attention import IncidenceAttention
from .mobius_layer import MobiusLayer
from .nested_optimizer import NestedOptimizer
from .hope_integration import HopeIntegration

__all__ = [
    'SpinorEmbedding',
    'IncidenceAttention',
    'MobiusLayer',
    'NestedOptimizer',
    'HopeIntegration',
]

__version__ = '0.1.0'

