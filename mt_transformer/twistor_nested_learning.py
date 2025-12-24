"""
扭量嵌套学习 (Twistor Nested Learning)
将HOPE的Nested Learning迁移到扭量几何框架
支持动态嵌套权重、层级学习率策略、残差连接优化
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


class DynamicNestedWeights(nn.Module):
    """
    动态嵌套权重调整：根据训练进度动态调整权重
    权重从初始值逐渐增加到目标值（模拟warmup）
    """
    
    def __init__(
        self,
        num_levels: int,
        initial_weight: float = 0.01,
        target_weight: float = 0.1
    ):
        super().__init__()
        self.num_levels = num_levels
        self.initial_weight = initial_weight
        self.target_weight = target_weight
        
        # 可学习的权重调整因子
        self.weight_factors = nn.Parameter(
            torch.ones(num_levels) * initial_weight
        )
    
    def forward(self, current_step: int, total_steps: int) -> torch.Tensor:
        """
        根据训练进度调整权重
        
        Args:
            current_step: 当前训练步数
            total_steps: 总训练步数
        
        Returns:
            调整后的权重张量，形状为 (num_levels,)
        """
        # 根据训练进度调整权重
        progress = min(current_step / max(total_steps, 1), 1.0)
        base_weights = self.initial_weight + \
                      (self.target_weight - self.initial_weight) * progress
        
        # 应用可学习的调整因子
        weights = base_weights * torch.sigmoid(self.weight_factors)
        
        return weights


class LevelWiseLearningRate:
    """
    层级学习率策略：深层使用较小学习率，浅层使用较大学习率
    实现学习率衰减策略
    """
    
    def __init__(
        self,
        base_lr: float,
        decay_factor: float = 0.9,
        num_levels: int = 5
    ):
        self.base_lr = base_lr
        self.decay_factor = decay_factor
        self.num_levels = num_levels
    
    def get_level_lrs(self) -> List[float]:
        """
        获取每层的学习率
        
        Returns:
            学习率列表，深层学习率 = base_lr * (decay_factor ** level)
        """
        return [
            self.base_lr * (self.decay_factor ** level)
            for level in range(self.num_levels)
        ]
    
    def get_level_lr(self, level: int) -> float:
        """获取指定层级的学习率"""
        return self.base_lr * (self.decay_factor ** level)


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
    新增功能：
    - 动态嵌套权重调整
    - 层级学习率策略
    - 可学习的残差连接权重
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
        use_level_constraints: bool = True,
        use_dynamic_weights: bool = True,
        use_levelwise_lr: bool = True,
        lr_decay_factor: float = 0.9,
        use_learnable_residual: bool = True,
        use_pipeline_parallel: bool = True,
        pipeline_stages: Optional[int] = None
    ):
        super().__init__()
        self.dim = dim
        self.omega_lr = omega_lr
        self.pi_lr = pi_lr
        self.constraint_weight = constraint_weight
        self.optimizer_type = optimizer_type
        self.num_nested_levels = num_nested_levels
        self.use_level_constraints = use_level_constraints
        self.use_dynamic_weights = use_dynamic_weights
        self.use_levelwise_lr = use_levelwise_lr
        self.use_learnable_residual = use_learnable_residual
        self.use_pipeline_parallel = use_pipeline_parallel
        self.pipeline_stages = pipeline_stages if pipeline_stages is not None else max(2, num_nested_levels // 2)
        
        # 层级学习率策略
        if use_levelwise_lr:
            omega_lr_strategy = LevelWiseLearningRate(omega_lr, lr_decay_factor, num_nested_levels)
            pi_lr_strategy = LevelWiseLearningRate(pi_lr, lr_decay_factor, num_nested_levels)
            self.omega_level_lrs = omega_lr_strategy.get_level_lrs()
            self.pi_level_lrs = pi_lr_strategy.get_level_lrs()
        elif nested_level_lrs is not None:
            # 使用指定的学习率
            if len(nested_level_lrs) != num_nested_levels:
                raise ValueError(f"nested_level_lrs长度必须等于num_nested_levels ({num_nested_levels})")
            self.omega_level_lrs = nested_level_lrs
            self.pi_level_lrs = nested_level_lrs
        else:
            # 默认：每层使用统一学习率
            self.omega_level_lrs = [omega_lr] * num_nested_levels
            self.pi_level_lrs = [pi_lr] * num_nested_levels
        
        # 嵌套网络：每层都是独立的网络模块
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
        
        # 动态嵌套权重（如果启用）
        if use_dynamic_weights:
            self.dynamic_omega_weights = DynamicNestedWeights(
                num_nested_levels, initial_weight=0.01, target_weight=0.1
            )
            self.dynamic_pi_weights = DynamicNestedWeights(
                num_nested_levels, initial_weight=0.01, target_weight=0.1
            )
        else:
            self.dynamic_omega_weights = None
            self.dynamic_pi_weights = None
        
        # 每层有自己的嵌套权重（递减权重）
        self.omega_nested_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1) * (0.1 / (i + 1))) 
            for i in range(num_nested_levels)
        ])
        
        self.pi_nested_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1) * (0.1 / (i + 1))) 
            for i in range(num_nested_levels)
        ])
        
        # 可学习的残差连接权重（如果启用）
        if use_learnable_residual:
            self.omega_residual_weights = nn.ParameterList([
                nn.Parameter(torch.ones(1) * 0.5)  # 初始权重0.5，平衡残差和变换
                for _ in range(num_nested_levels)
            ])
            self.pi_residual_weights = nn.ParameterList([
                nn.Parameter(torch.ones(1) * 0.5)
                for _ in range(num_nested_levels)
            ])
        else:
            self.omega_residual_weights = None
            self.pi_residual_weights = None
        
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
        
        # 训练步数跟踪（用于动态权重）
        self.register_buffer('current_step', torch.tensor(0))
        self.register_buffer('total_steps', torch.tensor(10000))  # 默认值
        
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
    
    def set_training_progress(self, current_step: int, total_steps: int):
        """
        设置训练进度（用于动态权重调整）
        
        Args:
            current_step: 当前训练步数
            total_steps: 总训练步数
        """
        self.current_step.fill_(current_step)
        self.total_steps.fill_(total_steps)
    
    def forward(
        self,
        x: torch.Tensor,
        compute_constraint: bool = True
    ) -> tuple:
        """
        前向传播：扭量嵌套学习（多层嵌套）
        支持动态权重、可学习残差连接、流水线并行
        
        Args:
            x: 输入张量（扭量表示），形状为 (batch, seq, dim)
            compute_constraint: 是否计算关联约束
        
        Returns:
            (output, constraint_loss): 输出张量和约束损失
        """
        # 如果启用流水线并行，使用流水线方式
        if self.use_pipeline_parallel:
            return self._forward_pipeline_parallel(x, compute_constraint)
        else:
            # 使用原有的顺序处理方式
            return self._forward_sequential(x, compute_constraint)
    
    def _forward_sequential(
        self,
        x: torch.Tensor,
        compute_constraint: bool = True
    ) -> tuple:
        """原有的顺序处理方式（向后兼容）"""
        # 分离ω和π分量
        omega, pi = x.chunk(2, dim=-1)  # 各为 (batch, seq, dim//2)
        
        # 保存初始值用于残差连接
        omega_input = omega
        pi_input = pi
        
        # 获取动态权重（如果启用）
        if self.use_dynamic_weights and self.dynamic_omega_weights is not None:
            dynamic_omega_ws = self.dynamic_omega_weights(
                int(self.current_step.item()),
                int(self.total_steps.item())
            )
            dynamic_pi_ws = self.dynamic_pi_weights(
                int(self.current_step.item()),
                int(self.total_steps.item())
            )
        else:
            dynamic_omega_ws = None
            dynamic_pi_ws = None
        
        # 逐层处理（嵌套前向传播）
        omega_prev = omega
        pi_prev = pi
        constraint_loss = torch.tensor(0.0, device=x.device)
        
        for level in range(self.num_nested_levels):
            # 通过当前层的网络处理
            omega_level = self.omega_nets[level](omega_prev)
            pi_level = self.pi_nets[level](pi_prev)
            
            # 应用嵌套权重（残差连接）
            if self.use_learnable_residual and self.omega_residual_weights is not None:
                # 使用可学习的残差权重
                residual_weight_omega = torch.sigmoid(self.omega_residual_weights[level])
                residual_weight_pi = torch.sigmoid(self.pi_residual_weights[level])
                
                omega_prev = residual_weight_omega * omega_prev + \
                            (1 - residual_weight_omega) * omega_level
                pi_prev = residual_weight_pi * pi_prev + \
                          (1 - residual_weight_pi) * pi_level
            else:
                # 使用固定嵌套权重
                nested_weight_omega = self.omega_nested_weights[level]
                nested_weight_pi = self.pi_nested_weights[level]
                
                # 如果使用动态权重，则应用动态调整
                if dynamic_omega_ws is not None:
                    nested_weight_omega = nested_weight_omega * dynamic_omega_ws[level]
                    nested_weight_pi = nested_weight_pi * dynamic_pi_ws[level]
                
                omega_prev = omega_prev + nested_weight_omega * (omega_level - omega_prev)
                pi_prev = pi_prev + nested_weight_pi * (pi_level - pi_prev)
            
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
    
    def _forward_pipeline_parallel(
        self,
        x: torch.Tensor,
        compute_constraint: bool = True
    ) -> tuple:
        """流水线并行处理方式"""
        # 分离ω和π分量
        omega, pi = x.chunk(2, dim=-1)  # 各为 (batch, seq, dim//2)
        
        # 获取动态权重（如果启用）
        if self.use_dynamic_weights and self.dynamic_omega_weights is not None:
            dynamic_omega_ws = self.dynamic_omega_weights(
                int(self.current_step.item()),
                int(self.total_steps.item())
            )
            dynamic_pi_ws = self.dynamic_pi_weights(
                int(self.current_step.item()),
                int(self.total_steps.item())
            )
        else:
            dynamic_omega_ws = None
            dynamic_pi_ws = None
        
        # 划分流水线阶段
        levels_per_stage = max(1, self.num_nested_levels // self.pipeline_stages)
        stages = []
        level_idx = 0
        while level_idx < self.num_nested_levels:
            stage_end = min(level_idx + levels_per_stage, self.num_nested_levels)
            stages.append((level_idx, stage_end))
            level_idx = stage_end
        
        # 流水线处理：每个阶段并行计算其内部的层级
        omega_prev = omega
        pi_prev = pi
        constraint_loss = torch.tensor(0.0, device=x.device)
        
        for stage_idx, (stage_start, stage_end) in enumerate(stages):
            # 当前阶段处理多个层级（阶段内顺序，但可以与其他阶段重叠）
            stage_omega = omega_prev
            stage_pi = pi_prev
            
            # 阶段内的层级处理（可以并行化，但为了简化先顺序处理）
            for level in range(stage_start, stage_end):
                # 通过当前层的网络处理
                omega_level = self.omega_nets[level](stage_omega)
                pi_level = self.pi_nets[level](stage_pi)
                
                # 应用嵌套权重（残差连接）
                if self.use_learnable_residual and self.omega_residual_weights is not None:
                    residual_weight_omega = torch.sigmoid(self.omega_residual_weights[level])
                    residual_weight_pi = torch.sigmoid(self.pi_residual_weights[level])
                    
                    stage_omega = residual_weight_omega * stage_omega + \
                                (1 - residual_weight_omega) * omega_level
                    stage_pi = residual_weight_pi * stage_pi + \
                              (1 - residual_weight_pi) * pi_level
                else:
                    nested_weight_omega = self.omega_nested_weights[level]
                    nested_weight_pi = self.pi_nested_weights[level]
                    
                    if dynamic_omega_ws is not None:
                        nested_weight_omega = nested_weight_omega * dynamic_omega_ws[level]
                        nested_weight_pi = nested_weight_pi * dynamic_pi_ws[level]
                    
                    stage_omega = stage_omega + nested_weight_omega * (omega_level - stage_omega)
                    stage_pi = stage_pi + nested_weight_pi * (pi_level - stage_pi)
                
                # 层级间的关联约束（如果启用）
                if self.use_level_constraints and level < self.num_nested_levels - 1:
                    level_constraint = self.level_constraints[level](stage_omega, stage_pi)
                    constraint_loss = constraint_loss + level_constraint
            
            # 更新为下一阶段的输入
            omega_prev = stage_omega
            pi_prev = stage_pi
        
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
                f'use_level_constraints={self.use_level_constraints}, '
                f'use_dynamic_weights={self.use_dynamic_weights}, '
                f'use_levelwise_lr={self.use_levelwise_lr}, '
                f'use_learnable_residual={self.use_learnable_residual}, '
                f'use_pipeline_parallel={self.use_pipeline_parallel}, '
                f'pipeline_stages={self.pipeline_stages}')

