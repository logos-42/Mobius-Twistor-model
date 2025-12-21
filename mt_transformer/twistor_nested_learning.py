"""
扭量嵌套学习 (Twistor Nested Learning)
将HOPE的Nested Learning迁移到扭量几何框架
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any, List


class OmegaNestedOptimizer:
    """
    ω分量的嵌套优化器
    在复射影空间中对ω分量进行独立优化
    """
    
    def __init__(
        self,
        parameters,
        optimizer_type: type = optim.Adam,
        lr: float = 1e-4,
        **optimizer_kwargs
    ):
        self.parameters = parameters
        self.optimizer = optimizer_type(parameters, lr=lr, **optimizer_kwargs)
        self.lr = lr
        
    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()
    
    def step(self):
        """执行优化步骤"""
        self.optimizer.step()
    
    def state_dict(self):
        """获取优化器状态"""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """加载优化器状态"""
        self.optimizer.load_state_dict(state_dict)


class PiNestedOptimizer:
    """
    π分量的嵌套优化器
    在复射影空间中对π分量进行独立优化
    """
    
    def __init__(
        self,
        parameters,
        optimizer_type: type = optim.Adam,
        lr: float = 1e-4,
        **optimizer_kwargs
    ):
        self.parameters = parameters
        self.optimizer = optimizer_type(parameters, lr=lr, **optimizer_kwargs)
        self.lr = lr
        
    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()
    
    def step(self):
        """执行优化步骤"""
        self.optimizer.step()
    
    def state_dict(self):
        """获取优化器状态"""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """加载优化器状态"""
        self.optimizer.load_state_dict(state_dict)


class IncidenceConstraint(nn.Module):
    """
    关联方程约束：通过扭量关联方程连接ω和π的嵌套优化层级
    
    扭量关联方程：ω · π* + π · ω* = 0 (在扭量空间中)
    这里我们使用一个可学习的约束项来近似这个关系
    """
    
    def __init__(self, dim: int, constraint_weight: float = 0.1):
        super().__init__()
        self.dim = dim
        self.constraint_weight = constraint_weight
        
        # 关联约束网络：学习ω和π之间的关联关系
        self.constraint_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )
        
    def forward(
        self,
        omega: torch.Tensor,
        pi: torch.Tensor
    ) -> torch.Tensor:
        """
        计算关联约束损失
        
        Args:
            omega: ω分量，形状为 (batch, seq, dim//2)
            pi: π分量，形状为 (batch, seq, dim//2)
        
        Returns:
            约束损失（标量）
        """
        # 组合ω和π
        twistor = torch.cat([omega, pi], dim=-1)  # (batch, seq, dim)
        
        # 计算关联约束
        constraint = self.constraint_net(twistor)  # (batch, seq, 1)
        
        # 约束应该接近0（关联方程）
        constraint_loss = constraint.abs().mean()
        
        return constraint_loss * self.constraint_weight


class TwistorNestedLearning(nn.Module):
    """
    扭量嵌套学习：整合ω/π嵌套优化器、关联方程约束
    
    将HOPE的Nested Learning迁移到扭量几何框架
    支持多层嵌套结构（默认5层）
    """
    
    def __init__(
        self,
        dim: int,
        omega_lr: float = 1e-4,
        pi_lr: float = 1e-4,
        constraint_weight: float = 0.1,
        optimizer_type: type = optim.Adam,
        num_nested_levels: int = 5,
        nested_level_lrs: Optional[List[float]] = None,
        use_level_constraints: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.omega_lr = omega_lr
        self.pi_lr = pi_lr
        self.constraint_weight = constraint_weight
        self.optimizer_type = optimizer_type
        self.num_nested_levels = num_nested_levels
        self.use_level_constraints = use_level_constraints
        
        # 设置每层的学习率
        if nested_level_lrs is None:
            # 默认：每层使用统一学习率
            self.omega_level_lrs = [omega_lr] * num_nested_levels
            self.pi_level_lrs = [pi_lr] * num_nested_levels
        else:
            # 使用指定的学习率
            if len(nested_level_lrs) != num_nested_levels:
                raise ValueError(f"nested_level_lrs长度必须等于num_nested_levels ({num_nested_levels})")
            self.omega_level_lrs = nested_level_lrs
            self.pi_level_lrs = nested_level_lrs
        
        # 5层嵌套网络：每层都是独立的网络模块
        self.omega_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim // 2, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, dim // 2)
            ) for _ in range(num_nested_levels)
        ])
        
        self.pi_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim // 2, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, dim // 2)
            ) for _ in range(num_nested_levels)
        ])
        
        # 每层有自己的嵌套权重（递减权重）
        self.omega_nested_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1) * (0.1 / (i + 1))) 
            for i in range(num_nested_levels)
        ])
        
        self.pi_nested_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1) * (0.1 / (i + 1))) 
            for i in range(num_nested_levels)
        ])
        
        # 创建嵌套优化器（每层一个）
        self.omega_optimizers: List[Optional[OmegaNestedOptimizer]] = [None] * num_nested_levels
        self.pi_optimizers: List[Optional[PiNestedOptimizer]] = [None] * num_nested_levels
        
        # 关联方程约束（全局）
        self.incidence_constraint = IncidenceConstraint(dim, constraint_weight)
        
        # 层级间的关联约束（如果启用）
        if use_level_constraints:
            self.level_constraints = nn.ModuleList([
                IncidenceConstraint(dim, constraint_weight * 0.5) 
                for _ in range(num_nested_levels - 1)
            ])
        else:
            self.level_constraints = None
        
    def create_nested_optimizers(self):
        """创建嵌套优化器（每层一个）"""
        for level in range(self.num_nested_levels):
            if self.omega_optimizers[level] is None:
                self.omega_optimizers[level] = OmegaNestedOptimizer(
                    self.omega_nets[level].parameters(),
                    optimizer_type=self.optimizer_type,
                    lr=self.omega_level_lrs[level]
                )
            
            if self.pi_optimizers[level] is None:
                self.pi_optimizers[level] = PiNestedOptimizer(
                    self.pi_nets[level].parameters(),
                    optimizer_type=self.optimizer_type,
                    lr=self.pi_level_lrs[level]
                )
    
    def forward(
        self,
        x: torch.Tensor,
        compute_constraint: bool = True
    ) -> tuple:
        """
        前向传播：扭量嵌套学习（多层嵌套）
        
        Args:
            x: 输入张量（扭量表示），形状为 (batch, seq, dim)
            compute_constraint: 是否计算关联约束
        
        Returns:
            (output, constraint_loss): 输出张量和约束损失
        """
        # 分离ω和π分量
        omega, pi = x.chunk(2, dim=-1)  # 各为 (batch, seq, dim//2)
        
        # 保存初始值用于残差连接
        omega_input = omega
        pi_input = pi
        
        # 逐层处理（嵌套前向传播）
        omega_prev = omega
        pi_prev = pi
        constraint_loss = torch.tensor(0.0, device=x.device)
        
        for level in range(self.num_nested_levels):
            # 通过当前层的网络处理
            omega_level = self.omega_nets[level](omega_prev)
            pi_level = self.pi_nets[level](pi_prev)
            
            # 应用嵌套权重（残差连接）
            omega_prev = omega_prev + self.omega_nested_weights[level] * (omega_level - omega_prev)
            pi_prev = pi_prev + self.pi_nested_weights[level] * (pi_level - pi_prev)
            
            # 层级间的关联约束（如果启用）
            if self.use_level_constraints and level < self.num_nested_levels - 1:
                level_constraint = self.level_constraints[level](omega_prev, pi_prev)
                constraint_loss = constraint_loss + level_constraint
        
        # 最终输出
        omega_processed = omega_prev
        pi_processed = pi_prev
        
        # 组合输出
        output = torch.cat([omega_processed, pi_processed], dim=-1)
        
        # 计算全局关联约束损失
        if compute_constraint:
            global_constraint = self.incidence_constraint(omega_processed, pi_processed)
            constraint_loss = constraint_loss + global_constraint
        
        return output, constraint_loss
    
    def nested_optimization_step(
        self,
        omega_losses: Optional[List[torch.Tensor]] = None,
        pi_losses: Optional[List[torch.Tensor]] = None,
        global_omega_loss: Optional[torch.Tensor] = None,
        global_pi_loss: Optional[torch.Tensor] = None
    ):
        """
        执行嵌套优化步骤（多层嵌套优化）
        
        Args:
            omega_losses: 每层ω分量的损失列表（可选）
            pi_losses: 每层π分量的损失列表（可选）
            global_omega_loss: 全局ω分量的损失（可选）
            global_pi_loss: 全局π分量的损失（可选）
        """
        if any(opt is None for opt in self.omega_optimizers) or any(opt is None for opt in self.pi_optimizers):
            self.create_nested_optimizers()
        
        # 如果提供了每层损失，逐层优化
        if omega_losses is not None and pi_losses is not None:
            for level in range(self.num_nested_levels):
                if level < len(omega_losses) and omega_losses[level] is not None:
                    self.omega_optimizers[level].zero_grad()
                    omega_losses[level].backward(retain_graph=True)
                    self.omega_optimizers[level].step()
                
                if level < len(pi_losses) and pi_losses[level] is not None:
                    self.pi_optimizers[level].zero_grad()
                    pi_losses[level].backward(retain_graph=True)
                    self.pi_optimizers[level].step()
        
        # 如果提供了全局损失，使用全局损失优化所有层
        if global_omega_loss is not None:
            for optimizer in self.omega_optimizers:
                optimizer.zero_grad()
            global_omega_loss.backward(retain_graph=True)
            for optimizer in self.omega_optimizers:
                optimizer.step()
        
        if global_pi_loss is not None:
            for optimizer in self.pi_optimizers:
                optimizer.zero_grad()
            global_pi_loss.backward()
            for optimizer in self.pi_optimizers:
                optimizer.step()
    
    def extra_repr(self) -> str:
        return (f'dim={self.dim}, omega_lr={self.omega_lr}, pi_lr={self.pi_lr}, '
                f'constraint_weight={self.constraint_weight}, '
                f'num_nested_levels={self.num_nested_levels}, '
                f'use_level_constraints={self.use_level_constraints}')

