"""
模型配置模块
定义不同规模的预设配置：small, medium, large, recommended
"""

from typing import Dict, Any, Optional


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    'small': {
        'dim': 128,
        'num_recurrent_layers': 2,
        'num_nested_levels': 5,
        'num_memories': 3,
        'bidirectional': False,
        'num_mobius_cycles': 3,
        'use_adaptive_evolution_rate': True,
        'use_multiscale_evolution': True,
        'use_level_constraints': True
    },
    'medium': {
        'dim': 192,
        'num_recurrent_layers': 2,
        'num_nested_levels': 5,
        'num_memories': 4,
        'bidirectional': False,
        'num_mobius_cycles': 3,
        'use_adaptive_evolution_rate': True,
        'use_multiscale_evolution': True,
        'use_level_constraints': True
    },
    'recommended': {
        'dim': 192,
        'num_recurrent_layers': 3,
        'num_nested_levels': 6,
        'num_memories': 5,
        'bidirectional': False,  # 暂时禁用双向以避免维度问题
        'num_mobius_cycles': 3,
        'use_adaptive_evolution_rate': True,
        'use_multiscale_evolution': True,
        'use_level_constraints': True
    },
    'large': {
        'dim': 512,
        'num_recurrent_layers': 4,
        'num_nested_levels': 8,
        'num_memories': 8,
        'bidirectional': True,
        'num_mobius_cycles': 3,
        'use_adaptive_evolution_rate': True,
        'use_multiscale_evolution': True,
        'use_level_constraints': True
    }
}


def get_config(config_name: str = 'recommended') -> Dict[str, Any]:
    """
    获取预设配置
    
    Args:
        config_name: 配置名称 ('small', 'medium', 'recommended', 'large')
    
    Returns:
        配置字典
    """
    if config_name not in MODEL_CONFIGS:
        raise ValueError(
            f"未知配置名称: {config_name}。"
            f"可用配置: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[config_name].copy()


def create_full_config(
    config_name: str = 'recommended',
    vocab_size: int = 1000,
    hidden_dim: Optional[int] = None,
    num_memory_cycles: int = 2,
    use_nested_learning: bool = True,
    use_phase_compression: bool = True,
    dropout: float = 0.1,
    nested_level_lrs: Optional[list] = None
) -> Dict[str, Any]:
    """
    创建完整的模型配置（包含所有必需参数）
    
    Args:
        config_name: 预设配置名称
        vocab_size: 词汇表大小
        hidden_dim: 隐藏维度（默认与dim相同）
        num_memory_cycles: 记忆循环次数
        use_nested_learning: 是否使用嵌套学习
        use_phase_compression: 是否使用相位压缩
        dropout: Dropout比率
        nested_level_lrs: 嵌套层级学习率列表
    
    Returns:
        完整配置字典
    """
    base_config = get_config(config_name)
    
    # 设置默认值
    if hidden_dim is None:
        hidden_dim = base_config['dim']
    
    # 构建完整配置
    full_config = {
        'vocab_size': vocab_size,
        'dim': base_config['dim'],
        'hidden_dim': hidden_dim,
        'num_recurrent_layers': base_config['num_recurrent_layers'],
        'num_memories': base_config['num_memories'],
        'num_memory_cycles': num_memory_cycles,
        'use_nested_learning': use_nested_learning,
        'use_phase_compression': use_phase_compression,
        'bidirectional': base_config['bidirectional'],
        'dropout': dropout,
        'num_mobius_cycles': base_config['num_mobius_cycles'],
        'use_adaptive_evolution_rate': base_config['use_adaptive_evolution_rate'],
        'use_multiscale_evolution': base_config['use_multiscale_evolution'],
        'num_nested_levels': base_config['num_nested_levels'],
        'nested_level_lrs': nested_level_lrs,
        'use_level_constraints': base_config['use_level_constraints']
    }
    
    return full_config


def estimate_parameters(config: Dict[str, Any]) -> int:
    """
    估算模型参数量（粗略估算）
    
    Args:
        config: 模型配置
    
    Returns:
        估算的参数量
    """
    vocab_size = config.get('vocab_size', 1000)
    dim = config['dim']
    num_recurrent_layers = config['num_recurrent_layers']
    num_memories = config['num_memories']
    num_nested_levels = config.get('num_nested_levels', 5)
    bidirectional = config.get('bidirectional', False)
    
    # 嵌入层
    embedding_params = vocab_size * dim
    
    # 循环层（粗略估算）
    # 每个循环层大约: 4 * (dim^2 + dim) （LSTM-like）
    recurrent_params = num_recurrent_layers * 4 * (dim * dim + dim)
    if bidirectional:
        recurrent_params *= 2
    
    # 记忆系统
    memory_params = num_memories * dim * 2
    
    # 嵌套学习（每层大约 dim^2）
    nested_params = num_nested_levels * dim * dim * 2  # omega 和 pi
    
    # 输出投影
    output_params = dim * 2 * dim
    
    total = embedding_params + recurrent_params + memory_params + nested_params + output_params
    
    return int(total)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置的有效性
    
    Args:
        config: 模型配置
    
    Returns:
        是否有效
    """
    required_keys = ['dim', 'num_recurrent_layers', 'num_nested_levels', 'num_memories']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置缺少必需键: {key}")
        
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"配置键 {key} 必须是正整数")
    
    if config['dim'] % 2 != 0:
        raise ValueError(f"dim 必须是偶数（因为需要分为 omega 和 pi 分量）")
    
    return True

