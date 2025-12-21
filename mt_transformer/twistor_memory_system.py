"""
扭量记忆系统 (Twistor Memory System)
将HOPE的Continuum Memory System迁移到扭量框架，利用莫比乌斯拓扑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .mobius_layer import MobiusLayer


class TwistorMemoryState(nn.Module):
    """
    扭量记忆状态：记忆以(ω_memory, π_memory)形式存储
    """
    
    def __init__(self, dim: int, num_memories: int = 1):
        super().__init__()
        self.dim = dim
        self.num_memories = num_memories
        
        # 扭量记忆状态：每个记忆都是(ω, π)对
        # 形状: (num_memories, dim)
        self.register_buffer(
            'omega_memory',
            torch.zeros(num_memories, dim // 2)
        )
        self.register_buffer(
            'pi_memory',
            torch.zeros(num_memories, dim // 2)
        )
        
        # 记忆重要性权重（可学习）
        self.memory_weights = nn.Parameter(torch.ones(num_memories) / num_memories)
        
    def get_memory(self) -> torch.Tensor:
        """
        获取当前记忆状态
        
        Returns:
            记忆状态，形状为 (num_memories, dim)
        """
        # 组合ω和π记忆
        memory = torch.cat([self.omega_memory, self.pi_memory], dim=-1)
        return memory
    
    def update_memory(
        self,
        omega_new: torch.Tensor,
        pi_new: torch.Tensor,
        update_rate: float = 0.1
    ):
        """
        更新记忆状态
        
        Args:
            omega_new: 新的ω分量，形状为 (batch, seq, dim//2) 或 (dim//2,)
            pi_new: 新的π分量，形状为 (batch, seq, dim//2) 或 (dim//2,)
            update_rate: 更新率
        """
        # 聚合输入（如果是序列，取均值）
        if omega_new.dim() == 3:
            omega_agg = omega_new.mean(dim=(0, 1))  # (dim//2,)
            pi_agg = pi_new.mean(dim=(0, 1))
        elif omega_new.dim() == 2:
            omega_agg = omega_new.mean(dim=0)  # (dim//2,)
            pi_agg = pi_new.mean(dim=0)
        else:
            omega_agg = omega_new
            pi_agg = pi_new
        
        # 更新记忆（使用移动平均）
        with torch.no_grad():
            self.omega_memory = (
                (1 - update_rate) * self.omega_memory +
                update_rate * omega_agg.unsqueeze(0).expand_as(self.omega_memory)
            )
            self.pi_memory = (
                (1 - update_rate) * self.pi_memory +
                update_rate * pi_agg.unsqueeze(0).expand_as(self.pi_memory)
            )


class MobiusMemoryCycle(nn.Module):
    """
    莫比乌斯记忆循环：通过莫比乌斯层实现记忆的拓扑循环
    """
    
    def __init__(self, dim: int, num_cycles: int = 1):
        super().__init__()
        self.dim = dim
        self.num_cycles = num_cycles
        
        # 多个莫比乌斯层用于记忆循环
        self.mobius_layers = nn.ModuleList([
            MobiusLayer(dim=dim, coupling_coeff=0.1 / (i + 1))
            for i in range(num_cycles)
        ])
        
    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        前向传播：通过莫比乌斯循环处理记忆
        
        Args:
            memory: 记忆状态，形状为 (num_memories, dim) 或 (batch, seq, dim)
        
        Returns:
            处理后的记忆，形状与输入相同
        """
        x = memory
        for mobius_layer in self.mobius_layers:
            x = mobius_layer(x)
        return x


class PhaseCompression(nn.Module):
    """
    相位压缩机制：利用莫比乌斯环特性，只保留相位信息
    """
    
    def __init__(self, dim: int, compression_ratio: float = 0.5):
        super().__init__()
        self.dim = dim
        self.compression_ratio = compression_ratio
        
        # 压缩网络：将完整状态压缩为相位信息
        compressed_dim = int(dim * compression_ratio)
        self.compress = nn.Linear(dim, compressed_dim)
        self.decompress = nn.Linear(compressed_dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：压缩和恢复相位信息
        
        Args:
            x: 输入张量，形状为 (batch, seq, dim) 或 (num_memories, dim)
        
        Returns:
            恢复后的张量，形状与输入相同
        """
        # 压缩：只保留相位信息
        phase = self.compress(x)
        
        # 恢复：从相位信息重建
        reconstructed = self.decompress(phase)
        
        return reconstructed


class TwistorMemorySystem(nn.Module):
    """
    扭量记忆系统：整合扭量记忆状态、莫比乌斯记忆循环、相位压缩
    
    将HOPE的CMS迁移到扭量框架
    """
    
    def __init__(
        self,
        dim: int,
        num_memories: int = 3,
        num_cycles: int = 2,
        update_rate: float = 0.1,
        use_phase_compression: bool = True,
        compression_ratio: float = 0.5
    ):
        super().__init__()
        self.dim = dim
        self.num_memories = num_memories
        self.update_rate = update_rate
        self.use_phase_compression = use_phase_compression
        
        # 扭量记忆状态
        self.memory_state = TwistorMemoryState(dim, num_memories)
        
        # 莫比乌斯记忆循环
        self.memory_cycle = MobiusMemoryCycle(dim, num_cycles)
        
        # 相位压缩（可选）
        if use_phase_compression:
            self.phase_compression = PhaseCompression(dim, compression_ratio)
        else:
            self.phase_compression = None
        
        # 记忆更新门控
        self.update_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_memories)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        update_memory: bool = True
    ) -> torch.Tensor:
        """
        前向传播：更新记忆并注入到输出
        
        Args:
            x: 输入张量，形状为 (batch, seq, dim)
            update_memory: 是否更新记忆（训练时使用）
        
        Returns:
            输出张量，形状与输入相同
        """
        batch_size, seq_len, dim = x.shape
        
        # 获取当前记忆状态
        memory = self.memory_state.get_memory()  # (num_memories, dim)
        
        # 通过莫比乌斯循环处理记忆
        processed_memory = self.memory_cycle(memory)  # (num_memories, dim)
        
        # 相位压缩（如果启用）
        if self.use_phase_compression and self.phase_compression is not None:
            processed_memory = self.phase_compression(processed_memory)
        
        # 计算记忆权重（使用门控机制）
        memory_weights = []
        for gate in self.update_gates:
            # 使用输入的均值作为上下文
            context = x.mean(dim=1)  # (batch, dim)
            weight = gate(context)  # (batch, 1)
            memory_weights.append(weight)
        
        # 组合记忆权重
        memory_weights = torch.stack(memory_weights, dim=1)  # (batch, num_memories, 1)
        memory_weights = F.softmax(memory_weights, dim=1)  # 归一化
        
        # 加权组合记忆
        # processed_memory: (num_memories, dim)
        # memory_weights: (batch, num_memories, 1)
        # 广播后: (batch, num_memories, dim)
        memory_broadcast = processed_memory.unsqueeze(0) * memory_weights  # (batch, num_memories, dim)
        memory_combined = memory_broadcast.sum(dim=1)  # (batch, dim)
        
        # 将记忆注入到输出
        memory_injected = memory_combined.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq, dim)
        output = x + memory_injected
        
        # 更新记忆（如果启用）
        if update_memory and self.training:
            # 分离ω和π分量
            omega, pi = x.chunk(2, dim=-1)
            # 使用加权更新率
            effective_update_rate = self.update_rate * memory_weights.mean()
            self.memory_state.update_memory(omega, pi, effective_update_rate)
        
        return output
    
    def get_memory_state(self) -> torch.Tensor:
        """
        获取当前记忆状态（用于可视化或分析）
        
        Returns:
            记忆状态，形状为 (num_memories, dim)
        """
        return self.memory_state.get_memory()
    
    def extra_repr(self) -> str:
        return (f'dim={self.dim}, num_memories={self.num_memories}, '
                f'update_rate={self.update_rate}, '
                f'use_phase_compression={self.use_phase_compression}')

