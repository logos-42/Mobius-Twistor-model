"""
扭量自我修正机制 - Recurrent版本 (Twistor Self-Modifying - Recurrent)
移除所有注意力机制，使用Recurrent结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .twistor_titans_recurrent import TwistorTitansRecurrent
from .mobius_layer import MobiusLayer


class AdaptiveMobiusLayer(nn.Module):
    """
    自适应莫比乌斯层：耦合系数可自我调整
    
    支持时间步级别的自适应和多次循环变换
    """
    
    def __init__(
        self,
        dim: int,
        base_coupling_coeff: float = 0.1,
        num_adaptive_cycles: int = 3,
        use_timestep_adaptive: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.base_coupling_coeff = base_coupling_coeff
        self.num_adaptive_cycles = num_adaptive_cycles
        self.use_timestep_adaptive = use_timestep_adaptive
        
        # 基础莫比乌斯层（用于单次变换）
        self.base_mobius = MobiusLayer(
            dim=dim,
            coupling_coeff=base_coupling_coeff,
            num_cycles=1,  # 单次循环，由外层控制多次
            progressive_coupling=False
        )
        
        # 增强的自适应耦合系数网络（3-4层）
        self.coupling_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 全局上下文网络（用于序列级别的自适应）
        self.global_context_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 自适应范围（可学习）
        self.adaptive_range = nn.Parameter(torch.ones(1) * 0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：自适应莫比乌斯变换（支持多次循环和时间步级别自适应）
        
        Args:
            x: 输入张量，形状为 (batch, seq, dim)
        
        Returns:
            输出张量，形状与输入相同
        """
        batch_size, seq_len, dim = x.shape
        output = x
        
        # 计算全局上下文（序列级别的特征）
        global_context = x.mean(dim=1, keepdim=True)  # (batch, 1, dim)
        global_coupling_factor = self.global_context_net(global_context)  # (batch, 1, 1)
        
        # 多次自适应莫比乌斯循环
        for cycle in range(self.num_adaptive_cycles):
            # 计算时间步级别的耦合系数（如果启用）
            if self.use_timestep_adaptive:
                # 对所有时间步并行计算
                timestep_coupling_factors = self.coupling_net(output)  # (batch, seq, 1)
                # 结合全局和局部上下文
                combined_factors = 0.7 * global_coupling_factor + 0.3 * timestep_coupling_factors
            else:
                # 只使用全局上下文，广播到所有时间步
                combined_factors = global_coupling_factor.expand(-1, seq_len, -1)  # (batch, seq, 1)
            
            # 动态调整耦合系数
            adaptive_coupling = (
                self.base_coupling_coeff + 
                self.adaptive_range * (combined_factors - 0.5) * 2
            )  # (batch, seq, 1)
            
            # 对每个时间步应用莫比乌斯变换
            cycle_output = []
            for t in range(seq_len):
                x_t = output[:, t:t+1, :]  # (batch, 1, dim)
                coupling_t = adaptive_coupling[:, t:t+1, :]  # (batch, 1, 1)
                
                # 执行莫比乌斯变换（手动实现，避免修改base_mobius）
                omega, pi = x_t.chunk(2, dim=-1)
                omega_dim = omega.shape[-1]
                
                if omega_dim % 2 == 0:
                    omega_real, omega_imag = omega.chunk(2, dim=-1)
                    pi_real, pi_imag = pi.chunk(2, dim=-1)
                    omega_new_real = pi_real
                    omega_new_imag = -pi_imag
                    pi_new_real = -omega_real
                    pi_new_imag = omega_imag
                    omega_new = torch.cat([omega_new_real, omega_new_imag], dim=-1)
                    pi_new = torch.cat([pi_new_real, pi_new_imag], dim=-1)
                else:
                    omega_new = pi
                    pi_new = -omega
                
                z_twisted = torch.cat([omega_new, pi_new], dim=-1)
                
                # 残差连接（使用自适应耦合系数）
                x_t_output = x_t + coupling_t * z_twisted
                cycle_output.append(x_t_output)
            
            # 堆叠所有时间步
            output = torch.cat(cycle_output, dim=1)  # (batch, seq, dim)
        
        return output


class TwistorSelfModifyingRecurrent(nn.Module):
    """
    扭量自我修正模块 - Recurrent版本
    
    完全移除注意力机制，使用TwistorTitansRecurrent实现循环结构。
    保持自我修正能力通过循环状态的自适应更新。
    
    Args:
        dim: 输入维度（扭量表示的维度）
        hidden_dim: 隐藏状态维度（默认与dim相同）
        num_layers: 循环层数（默认2）
        bidirectional: 是否使用双向循环（默认False）
        dropout: Dropout比率（默认0.1）
        use_mobius: 是否在循环单元中使用莫比乌斯变换（默认True）
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.1,
        use_mobius: bool = True,
        num_mobius_cycles: int = 3,
        use_adaptive_evolution_rate: bool = True,
        use_multiscale_evolution: bool = True,
        chunk_size: int = 512,
        use_chunk_parallel: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.use_mobius = use_mobius
        self.num_mobius_cycles = num_mobius_cycles
        self.use_adaptive_evolution_rate = use_adaptive_evolution_rate
        self.use_multiscale_evolution = use_multiscale_evolution
        self.chunk_size = chunk_size
        self.use_chunk_parallel = use_chunk_parallel
        
        # 扭量化Titans循环层（替代注意力机制）
        self.recurrent = TwistorTitansRecurrent(
            input_dim=dim,
            hidden_dim=self.hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            use_mobius=use_mobius,
            num_mobius_cycles=num_mobius_cycles,
            use_adaptive_evolution_rate=use_adaptive_evolution_rate,
            use_multiscale_evolution=use_multiscale_evolution,
            chunk_size=chunk_size,
            use_chunk_parallel=use_chunk_parallel
        )
        
        # 输出维度调整
        output_dim = self.hidden_dim * 2 if bidirectional else self.hidden_dim
        if output_dim != dim:
            self.output_proj = nn.Linear(output_dim, dim)
        else:
            self.output_proj = nn.Identity()
        
        # 自适应莫比乌斯层（用于拓扑变换）
        self.adaptive_mobius = AdaptiveMobiusLayer(
            dim=dim,
            num_adaptive_cycles=3,
            use_timestep_adaptive=True
        )
        
        # 前馈网络（可选，用于进一步处理）
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # 自我修正权重（全局）
        self.self_modify_weight = nn.Parameter(torch.ones(1) * 0.01)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：扭量自我修正（Recurrent版本）
        
        Args:
            x: 输入张量（扭量表示），形状为 (batch, seq, dim)
        
        Returns:
            输出张量，形状与输入相同
        """
        # 1. 残差连接 + 扭量化循环层（替代注意力）
        residual = x
        x = self.norm1(x)
        
        # 通过循环层处理
        x, _ = self.recurrent(x)  # (batch, seq, hidden_dim or hidden_dim*2)
        
        # 投影回原始维度
        x = self.output_proj(x)  # (batch, seq, dim)
        
        x = residual + x
        
        # 2. 残差连接 + 自适应莫比乌斯层
        residual = x
        x = self.norm2(x)
        x = self.adaptive_mobius(x)
        x = residual + x
        
        # 3. 残差连接 + 前馈网络
        residual = x
        x = self.ffn(x)
        x = residual + x
        
        # 4. 全局自我修正调整
        if self.training:
            modify_factor = torch.sigmoid(self.self_modify_weight)
            x = x * (1 + modify_factor * 0.1)
        
        return x
    
    def extra_repr(self) -> str:
        return (f'dim={self.dim}, hidden_dim={self.hidden_dim}, '
                f'num_layers={self.num_layers}, bidirectional={self.bidirectional}, '
                f'use_mobius={self.use_mobius}, chunk_size={self.chunk_size}, '
                f'use_chunk_parallel={self.use_chunk_parallel}')

