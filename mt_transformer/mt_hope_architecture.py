"""
MT-HOPE完整架构
整合所有组件，创建完整的扭量莫比乌斯-HOPE架构
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from .spinor_embedding import SpinorEmbedding
from .twistor_self_modifying import TwistorSelfModifying
from .twistor_memory_system import TwistorMemorySystem
from .twistor_nested_learning import TwistorNestedLearning


class MTHopeArchitecture(nn.Module):
    """
    MT-HOPE完整架构
    
    架构流程：
    Token IDs
      → SpinorEmbedding (扭量表示: ω, π)
      → TwistorSelfModifying (扭量自我修正)
        → AdaptiveIncidenceAttention (自适应关联注意力)
        → AdaptiveMobiusLayer (自适应莫比乌斯层)
      → TwistorMemorySystem (扭量记忆系统)
        → MobiusMemoryCycle (莫比乌斯记忆循环)
        → PhaseCompression (相位压缩)
      → TwistorNestedLearning (扭量嵌套学习)
        → OmegaNestedOptimizer (ω嵌套优化)
        → PiNestedOptimizer (π嵌套优化)
      → 输出
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_self_modifying_layers: int = 2,
        num_memories: int = 3,
        num_memory_cycles: int = 2,
        use_nested_learning: bool = True,
        use_phase_compression: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_self_modifying_layers = num_self_modifying_layers
        self.use_nested_learning = use_nested_learning
        self.use_phase_compression = use_phase_compression
        
        # 1. 旋量嵌入层：将Token映射为扭量表示
        self.embedding = SpinorEmbedding(
            vocab_size=vocab_size,
            dim=dim // 2,  # 每个分量的维度，总输出是dim
            use_complex=False
        )
        
        # 2. 扭量自我修正模块
        self.self_modifying = TwistorSelfModifying(
            dim=dim,  # 扭量表示的维度（ω和π各dim//2）
            num_layers=num_self_modifying_layers,
            dropout=dropout,
            use_evolution=True,
            use_adaptive_attention=True,
            use_adaptive_mobius=True
        )
        
        # 3. 扭量记忆系统
        self.memory_system = TwistorMemorySystem(
            dim=dim,
            num_memories=num_memories,
            num_cycles=num_memory_cycles,
            update_rate=0.1,
            use_phase_compression=use_phase_compression,
            compression_ratio=0.5
        )
        
        # 4. 扭量嵌套学习（可选）
        if use_nested_learning:
            self.nested_learning = TwistorNestedLearning(
                dim=dim,
                omega_lr=1e-4,
                pi_lr=1e-4,
                constraint_weight=0.1
            )
        else:
            self.nested_learning = None
        
        # 5. 输出投影（将扭量表示投影回标准维度）
        self.output_proj = nn.Linear(dim, dim)
        
        # 6. 最终归一化
        self.final_norm = nn.LayerNorm(dim)
        
    def forward(
        self,
        token_ids: torch.Tensor,
        return_constraint_loss: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播：完整的MT-HOPE架构流程
        
        Args:
            token_ids: Token ID张量，形状为 (batch_size, seq_len)
            return_constraint_loss: 是否返回关联约束损失
        
        Returns:
            输出张量，形状为 (batch_size, seq_len, dim)
            如果return_constraint_loss=True，还返回约束损失
        """
        # 1. 旋量嵌入：Token → 扭量表示
        x = self.embedding(token_ids)  # (batch, seq, dim)
        
        # 2. 扭量自我修正
        x = self.self_modifying(x)  # (batch, seq, dim)
        
        # 3. 扭量记忆系统
        x = self.memory_system(x, update_memory=self.training)  # (batch, seq, dim)
        
        # 4. 扭量嵌套学习（如果启用）
        constraint_loss = torch.tensor(0.0, device=x.device)
        if self.use_nested_learning and self.nested_learning is not None:
            x, constraint_loss = self.nested_learning(x, compute_constraint=True)
        
        # 5. 输出投影
        output = self.output_proj(x)  # (batch, seq, dim)
        
        # 6. 最终归一化
        output = self.final_norm(output)
        
        if return_constraint_loss:
            return output, constraint_loss
        
        return output
    
    def get_memory_state(self) -> torch.Tensor:
        """
        获取当前记忆状态（用于可视化或分析）
        
        Returns:
            记忆状态，形状为 (num_memories, dim)
        """
        return self.memory_system.get_memory_state()
    
    def create_nested_optimizers(self):
        """创建嵌套优化器（如果使用嵌套学习）"""
        if self.use_nested_learning and self.nested_learning is not None:
            self.nested_learning.create_nested_optimizers()
    
    def extra_repr(self) -> str:
        return (f'vocab_size={self.vocab_size}, dim={self.dim}, '
                f'num_self_modifying_layers={self.num_self_modifying_layers}, '
                f'use_nested_learning={self.use_nested_learning}, '
                f'use_phase_compression={self.use_phase_compression}')

