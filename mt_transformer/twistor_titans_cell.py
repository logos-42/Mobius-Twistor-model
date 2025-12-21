"""
扭量化Titans循环单元 (Twistor Titans Cell)
基于扭量演化的循环状态更新方程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AdaptiveEvolutionRate(nn.Module):
    """
    自适应演化率网络：根据上下文动态调整扭量演化速率
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.min_rate = 0.001
        self.max_rate = 0.1
    
    def forward(self, cell_state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        计算自适应演化率
        
        Args:
            cell_state: 细胞状态，形状为 (batch, hidden_dim)
            context: 上下文信息，形状为 (batch, hidden_dim)
        
        Returns:
            演化率，形状为 (batch, 1)
        """
        combined = torch.cat([cell_state, context], dim=-1)
        rate_factor = self.net(combined)
        return self.min_rate + (self.max_rate - self.min_rate) * rate_factor


class TwistorTitansCell(nn.Module):
    """
    扭量化Titans循环单元
    
    基于扭量演化的状态更新方程：
    h_t = f(ω_t, π_t, h_{t-1})
    
    其中：
    - h_t: 当前隐藏状态（扭量形式：ω_h, π_h）
    - ω_t, π_t: 当前输入（扭量表示）
    - h_{t-1}: 前一步隐藏状态
    - f: 扭量演化函数（包含莫比乌斯变换）
    
    Args:
        input_dim: 输入维度（扭量表示的维度，包含ω和π）
        hidden_dim: 隐藏状态维度（扭量形式的隐藏状态维度）
        use_mobius: 是否在状态更新中使用莫比乌斯变换（默认True）
        num_mobius_cycles: 莫比乌斯循环次数（默认3）
        use_adaptive_evolution_rate: 是否使用自适应演化率（默认True）
        use_multiscale_evolution: 是否使用多尺度演化（默认True）
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_mobius: bool = True,
        num_mobius_cycles: int = 3,
        use_adaptive_evolution_rate: bool = True,
        use_multiscale_evolution: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_mobius = use_mobius
        self.num_mobius_cycles = num_mobius_cycles
        self.use_adaptive_evolution_rate = use_adaptive_evolution_rate
        self.use_multiscale_evolution = use_multiscale_evolution
        
        # 输入门：控制输入如何影响状态
        # 输入：当前输入(ω_t, π_t) + 前一步隐藏状态(ω_h, π_h)
        self.input_gate_omega = nn.Linear(input_dim + hidden_dim, hidden_dim // 2)
        self.input_gate_pi = nn.Linear(input_dim + hidden_dim, hidden_dim // 2)
        
        # 遗忘门：控制状态如何更新
        self.forget_gate_omega = nn.Linear(input_dim + hidden_dim, hidden_dim // 2)
        self.forget_gate_pi = nn.Linear(input_dim + hidden_dim, hidden_dim // 2)
        
        # 候选值：新的状态候选
        self.candidate_omega = nn.Linear(input_dim + hidden_dim, hidden_dim // 2)
        self.candidate_pi = nn.Linear(input_dim + hidden_dim, hidden_dim // 2)
        
        # 输出门：控制输出
        self.output_gate_omega = nn.Linear(input_dim + hidden_dim, hidden_dim // 2)
        self.output_gate_pi = nn.Linear(input_dim + hidden_dim, hidden_dim // 2)
        
        # 扭量演化网络：多尺度演化
        if use_multiscale_evolution:
            # 快速演化：浅层网络，快速响应
            self.evolution_net_fast = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            )
            
            # 中速演化：中等深度
            self.evolution_net_medium = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh()
            )
            
            # 慢速演化：深层网络，长期模式
            self.evolution_net_slow = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh()
            )
            
            # 多尺度权重（可学习）
            self.scale_weights = nn.Parameter(torch.ones(3) / 3)  # [fast, medium, slow]
        else:
            # 单尺度演化网络（向后兼容）
            self.evolution_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh()
            )
        
        # 自适应演化率网络
        if use_adaptive_evolution_rate:
            self.adaptive_evolution_rate = AdaptiveEvolutionRate(hidden_dim)
        else:
            # 固定演化率（向后兼容）
            self.evolution_rate = nn.Parameter(torch.ones(1) * 0.01)
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[tuple] = None
    ) -> tuple:
        """
        前向传播：循环单元状态更新
        
        Args:
            x: 当前输入，形状为 (batch, input_dim)
                包含扭量表示(ω_t, π_t)
            hidden: 前一步隐藏状态，tuple of (omega_h, pi_h)
                每个形状为 (batch, hidden_dim // 2)
                如果为None，初始化为零
        
        Returns:
            (output, new_hidden): 
            - output: 当前输出，形状为 (batch, hidden_dim)
            - new_hidden: 新的隐藏状态，tuple of (omega_h_new, pi_h_new)
        """
        batch_size = x.shape[0]
        
        # 初始化隐藏状态（如果为None）
        if hidden is None:
            omega_h = torch.zeros(batch_size, self.hidden_dim // 2, device=x.device, dtype=x.dtype)
            pi_h = torch.zeros(batch_size, self.hidden_dim // 2, device=x.device, dtype=x.dtype)
        else:
            omega_h, pi_h = hidden
        
        # 组合隐藏状态
        h_prev = torch.cat([omega_h, pi_h], dim=-1)  # (batch, hidden_dim)
        
        # 组合输入和隐藏状态
        combined = torch.cat([x, h_prev], dim=-1)  # (batch, input_dim + hidden_dim)
        
        # 输入门：控制输入如何影响状态
        i_omega = torch.sigmoid(self.input_gate_omega(combined))
        i_pi = torch.sigmoid(self.input_gate_pi(combined))
        
        # 遗忘门：控制状态如何更新
        f_omega = torch.sigmoid(self.forget_gate_omega(combined))
        f_pi = torch.sigmoid(self.forget_gate_pi(combined))
        
        # 候选值：新的状态候选
        c_tilde_omega = torch.tanh(self.candidate_omega(combined))
        c_tilde_pi = torch.tanh(self.candidate_pi(combined))
        
        # 更新细胞状态（类似LSTM，但使用扭量形式）
        # 分离输入的ω和π
        omega_x, pi_x = x.chunk(2, dim=-1)
        
        # 更新ω和π状态
        omega_cell = f_omega * omega_h + i_omega * c_tilde_omega
        pi_cell = f_pi * pi_h + i_pi * c_tilde_pi
        
        # 扭量演化：ω和π的相互演化
        cell_state = torch.cat([omega_cell, pi_cell], dim=-1)  # (batch, hidden_dim)
        
        # 多尺度演化或单尺度演化
        if self.use_multiscale_evolution:
            # 计算三种尺度的演化
            evolution_fast = self.evolution_net_fast(cell_state)
            evolution_medium = self.evolution_net_medium(cell_state)
            evolution_slow = self.evolution_net_slow(cell_state)
            
            # 归一化权重
            scale_weights_norm = F.softmax(self.scale_weights, dim=0)
            
            # 组合多尺度演化
            evolution = (
                scale_weights_norm[0] * evolution_fast +
                scale_weights_norm[1] * evolution_medium +
                scale_weights_norm[2] * evolution_slow
            )
        else:
            # 单尺度演化（向后兼容）
            evolution = self.evolution_net(cell_state)
        
        evolution_omega, evolution_pi = evolution.chunk(2, dim=-1)
        
        # 自适应演化率或固定演化率
        if self.use_adaptive_evolution_rate:
            # 使用上下文（当前输入）作为上下文信息
            context = h_prev  # 使用前一步隐藏状态作为上下文
            evolution_rate = self.adaptive_evolution_rate(cell_state, context)
        else:
            # 固定演化率（向后兼容）
            evolution_rate = self.evolution_rate
        
        # 应用演化
        omega_cell = omega_cell + evolution_rate * evolution_omega
        pi_cell = pi_cell + evolution_rate * evolution_pi
        
        # 多层莫比乌斯变换（如果启用）
        if self.use_mobius:
            # 保存原始状态用于残差连接
            omega_cell_orig = omega_cell.clone()
            pi_cell_orig = pi_cell.clone()
            
            # 多次莫比乌斯循环
            for cycle in range(self.num_mobius_cycles):
                # 莫比乌斯变换：交换位置并取共轭
                omega_cell_new = pi_cell
                pi_cell_new = -omega_cell
                
                # 渐进式耦合：每次循环的耦合系数递增
                coupling_coeff = 0.1 * (cycle + 1) / self.num_mobius_cycles
                
                # 混合原始和变换后的状态
                omega_cell = (1 - coupling_coeff) * omega_cell + coupling_coeff * omega_cell_new
                pi_cell = (1 - coupling_coeff) * pi_cell + coupling_coeff * pi_cell_new
        
        # 输出门：控制输出
        o_omega = torch.sigmoid(self.output_gate_omega(combined))
        o_pi = torch.sigmoid(self.output_gate_pi(combined))
        
        # 计算输出
        omega_h_new = o_omega * torch.tanh(omega_cell)
        pi_h_new = o_pi * torch.tanh(pi_cell)
        
        # 组合输出
        output = torch.cat([omega_h_new, pi_h_new], dim=-1)  # (batch, hidden_dim)
        
        # 新的隐藏状态
        new_hidden = (omega_h_new, pi_h_new)
        
        return output, new_hidden
    
    def extra_repr(self) -> str:
        return (f'input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, '
                f'use_mobius={self.use_mobius}, num_mobius_cycles={self.num_mobius_cycles}, '
                f'use_adaptive_evolution_rate={self.use_adaptive_evolution_rate}, '
                f'use_multiscale_evolution={self.use_multiscale_evolution}')

