"""
嵌套优化器包装 (Nested Optimizer)
支持将MT组件作为嵌套优化层级，符合HOPE架构的Nested Learning范式
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Dict, Any


class NestedOptimizer(nn.Module):
    """
    嵌套优化器包装：将MT组件包装为嵌套优化问题的一个层级
    
    基于HOPE架构的Nested Learning范式，每个组件都可以作为
    嵌套优化问题的一个层级，有自己的"context flow"。
    
    Args:
        module: 要包装的模块（MT组件）
        inner_optimizer: 内部优化器类型（如torch.optim.Adam）
        inner_lr: 内部优化器的学习率（默认1e-4）
        context_flow_dim: 上下文流的维度（默认None，使用模块输出维度）
    """
    
    def __init__(
        self,
        module: nn.Module,
        inner_optimizer: type = torch.optim.Adam,
        inner_lr: float = 1e-4,
        context_flow_dim: Optional[int] = None
    ):
        super().__init__()
        self.module = module
        self.inner_optimizer_type = inner_optimizer
        self.inner_lr = inner_lr
        
        # 确定上下文流维度
        if context_flow_dim is None:
            # 尝试从模块获取输出维度
            if hasattr(module, 'dim'):
                context_flow_dim = module.dim
            else:
                # 默认值
                context_flow_dim = 512
        
        self.context_flow_dim = context_flow_dim
        
        # 上下文流状态（用于嵌套优化）
        # 这代表该层级的"记忆"或"状态"
        self.register_buffer('context_state', torch.zeros(1, context_flow_dim))
        
        # 内部优化器（在训练时创建）
        self.inner_optimizer: Optional[torch.optim.Optimizer] = None
        
    def forward(
        self,
        x: torch.Tensor,
        update_context: bool = False
    ) -> torch.Tensor:
        """
        前向传播：执行嵌套优化
        
        Args:
            x: 输入张量
            update_context: 是否更新上下文流（训练时使用）
        
        Returns:
            输出张量
        """
        # 将上下文状态与输入结合
        # 这里简化处理：将上下文状态广播到批次大小
        batch_size = x.shape[0]
        context = self.context_state.expand(batch_size, -1)
        
        # 如果输入是2D，需要添加序列维度
        if x.dim() == 2:
            # (batch, dim) -> (batch, 1, dim)
            x = x.unsqueeze(1)
            context = context.unsqueeze(1)
        
        # 将上下文注入输入（简化：拼接或相加）
        # 这里使用相加的方式
        if x.shape[-1] == self.context_flow_dim:
            x_with_context = x + context
        else:
            # 如果维度不匹配，使用线性投影
            if not hasattr(self, 'context_proj'):
                self.context_proj = nn.Linear(
                    self.context_flow_dim,
                    x.shape[-1],
                    device=x.device,
                    dtype=x.dtype
                )
            context_proj = self.context_proj(context)
            x_with_context = x + context_proj
        
        # 通过模块前向传播
        output = self.module(x_with_context)
        
        # 如果更新上下文，更新上下文状态
        if update_context and self.training:
            # 使用输出的某种聚合来更新上下文
            # 这里使用输出的均值
            if output.dim() == 3:
                # (batch, seq, dim) -> (batch, dim)
                output_agg = output.mean(dim=1)
            else:
                output_agg = output
            
            # 更新上下文状态（使用移动平均）
            with torch.no_grad():
                self.context_state = 0.9 * self.context_state + 0.1 * output_agg.mean(dim=0, keepdim=True)
        
        return output
    
    def create_inner_optimizer(self):
        """创建内部优化器（用于嵌套优化）"""
        if self.inner_optimizer is None:
            self.inner_optimizer = self.inner_optimizer_type(
                self.module.parameters(),
                lr=self.inner_lr
            )
        return self.inner_optimizer
    
    def step_inner_optimizer(self, loss: torch.Tensor):
        """
        执行内部优化器的一步
        
        Args:
            loss: 损失值
        """
        if self.inner_optimizer is None:
            self.create_inner_optimizer()
        
        self.inner_optimizer.zero_grad()
        loss.backward()
        self.inner_optimizer.step()
    
    def extra_repr(self) -> str:
        return f'context_flow_dim={self.context_flow_dim}, inner_lr={self.inner_lr}'

