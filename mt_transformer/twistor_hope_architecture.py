"""
扭量化HOPE完整架构 (Twistor HOPE Architecture)
真正的Recurrent结构，完全移除注意力机制
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from .spinor_embedding import SpinorEmbedding
from .twistor_self_modifying_recurrent import TwistorSelfModifyingRecurrent
from .twistor_memory_system import TwistorMemorySystem
from .twistor_nested_learning import TwistorNestedLearning


class TwistorHopeArchitecture(nn.Module):
    """
    扭量化HOPE完整架构 - Recurrent版本
    
    完全移除注意力机制，使用真正的Recurrent结构（基于Titans）。
    
    架构流程：
    Token IDs
      → SpinorEmbedding (扭量表示: ω, π)
      → TwistorSelfModifyingRecurrent (扭量自我修正，Recurrent版本)
        → TwistorTitansRecurrent (扭量化循环层)
        → AdaptiveMobiusLayer (自适应莫比乌斯层)
      → TwistorMemorySystem (扭量记忆系统，循环更新)
      → TwistorNestedLearning (扭量嵌套学习)
      → 输出
    
    Args:
        vocab_size: 词汇表大小
        dim: 模型维度
        hidden_dim: 循环层隐藏状态维度（默认与dim相同）
        num_recurrent_layers: 循环层数（默认2）
        num_memories: 记忆数量（默认3）
        num_memory_cycles: 记忆循环次数（默认2）
        use_nested_learning: 是否使用嵌套学习（默认True）
        use_phase_compression: 是否使用相位压缩（默认True）
        bidirectional: 是否使用双向循环（默认False）
        dropout: Dropout比率（默认0.1）
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_recurrent_layers: int = 2,
        num_memories: int = 3,
        num_memory_cycles: int = 2,
        use_nested_learning: bool = True,
        use_phase_compression: bool = True,
        bidirectional: bool = False,
        dropout: float = 0.1,
        # 新增参数：扭量演化相关
        num_mobius_cycles: int = 3,
        use_adaptive_evolution_rate: bool = True,
        use_multiscale_evolution: bool = True,
        # 新增参数：嵌套学习相关
        num_nested_levels: int = 5,
        nested_level_lrs: Optional[list] = None,
        use_level_constraints: bool = True,
        # 新增参数：并行优化相关
        chunk_size: int = 512,
        use_chunk_parallel: bool = True,
        use_pipeline_parallel: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else dim
        self.num_recurrent_layers = num_recurrent_layers
        self.use_nested_learning = use_nested_learning
        self.use_phase_compression = use_phase_compression
        self.bidirectional = bidirectional
        self.num_mobius_cycles = num_mobius_cycles
        self.use_adaptive_evolution_rate = use_adaptive_evolution_rate
        self.use_multiscale_evolution = use_multiscale_evolution
        self.num_nested_levels = num_nested_levels
        self.chunk_size = chunk_size
        self.use_chunk_parallel = use_chunk_parallel
        self.use_pipeline_parallel = use_pipeline_parallel
        
        # 1. 旋量嵌入层：将Token映射为扭量表示
        # 注意：SpinorEmbedding 的输出维度是 dim*2（因为 omega 和 pi 各占 dim）
        # 但每个分量内部又分为实部虚部，所以实际是 dim*2
        self.embedding = SpinorEmbedding(
            vocab_size=vocab_size,
            dim=dim // 2,  # 每个分量的维度，但输出是 dim*2（因为实部虚部分离）
            use_complex=False
        )
        
        # 2. 扭量自我修正模块（Recurrent版本，无注意力）
        # 注意：embedding 输出是 dim*2（omega 和 pi 各 dim），拼接后是 dim*2
        # 但实际上应该是 dim（如果按注释理解），我们需要检查实际输出
        # 临时修复：使用 dim*2
        self.self_modifying = TwistorSelfModifyingRecurrent(
            dim=dim * 2,  # 扭量表示的维度（omega 和 pi 拼接后）
            hidden_dim=self.hidden_dim * 2,
            num_layers=num_recurrent_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            use_mobius=True,
            num_mobius_cycles=num_mobius_cycles,
            use_adaptive_evolution_rate=use_adaptive_evolution_rate,
            use_multiscale_evolution=use_multiscale_evolution,
            chunk_size=chunk_size,
            use_chunk_parallel=use_chunk_parallel
        )
        
        # 3. 扭量记忆系统（循环更新版本）
        self.memory_system = TwistorMemorySystem(
            dim=dim * 2,
            num_memories=num_memories,
            num_cycles=num_memory_cycles,
            update_rate=0.1,
            use_phase_compression=use_phase_compression,
            compression_ratio=0.5
        )
        
        # 4. 扭量嵌套学习（可选）
        if use_nested_learning:
            self.nested_learning = TwistorNestedLearning(
                dim=dim * 2,
                omega_lr=1e-4,
                pi_lr=1e-4,
                constraint_weight=0.1,
                num_nested_levels=num_nested_levels,
                nested_level_lrs=nested_level_lrs,
                use_level_constraints=use_level_constraints,
                use_pipeline_parallel=use_pipeline_parallel
            )
        else:
            self.nested_learning = None
        
        # 5. 输出投影（将扭量表示投影回标准维度）
        self.output_proj = nn.Linear(dim * 2, dim)
        
        # 6. 最终归一化
        self.final_norm = nn.LayerNorm(dim)
        
    def forward(
        self,
        token_ids: torch.Tensor,
        return_constraint_loss: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播：完整的扭量化HOPE架构流程（Recurrent版本）
        
        Args:
            token_ids: Token ID张量，形状为 (batch_size, seq_len)
            return_constraint_loss: 是否返回关联约束损失
        
        Returns:
            输出张量，形状为 (batch_size, seq_len, dim)
            如果return_constraint_loss=True，还返回约束损失
        """
        # 1. 旋量嵌入：Token → 扭量表示
        x = self.embedding(token_ids)  # (batch, seq, dim)
        
        # 2. 扭量自我修正（Recurrent，无注意力）
        x = self.self_modifying(x)  # (batch, seq, dim)
        
        # 3. 扭量记忆系统（循环更新）
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
                f'hidden_dim={self.hidden_dim}, num_recurrent_layers={self.num_recurrent_layers}, '
                f'use_nested_learning={self.use_nested_learning}, '
                f'use_phase_compression={self.use_phase_compression}, '
                f'bidirectional={self.bidirectional}, '
                f'num_mobius_cycles={self.num_mobius_cycles}, '
                f'use_adaptive_evolution_rate={self.use_adaptive_evolution_rate}, '
                f'use_multiscale_evolution={self.use_multiscale_evolution}, '
                f'num_nested_levels={self.num_nested_levels}, '
                f'chunk_size={self.chunk_size}, use_chunk_parallel={self.use_chunk_parallel}, '
                f'use_pipeline_parallel={self.use_pipeline_parallel}')

