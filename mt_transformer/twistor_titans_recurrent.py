"""
扭量化Titans循环层 (Twistor Titans Recurrent)
使用TwistorTitansCell逐时间步处理序列
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from .twistor_titans_cell import TwistorTitansCell


class TwistorTitansRecurrent(nn.Module):
    """
    扭量化Titans循环层
    
    使用TwistorTitansCell逐时间步处理序列，实现真正的Recurrent结构。
    
    Args:
        input_dim: 输入维度（扭量表示的维度）
        hidden_dim: 隐藏状态维度（默认与input_dim相同）
        num_layers: 循环层数（默认1）
        bidirectional: 是否使用双向循环（默认False）
        dropout: Dropout比率（默认0.0）
        use_mobius: 是否在循环单元中使用莫比乌斯变换（默认True）
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        use_mobius: bool = True,
        num_mobius_cycles: int = 3,
        use_adaptive_evolution_rate: bool = True,
        use_multiscale_evolution: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_mobius_cycles = num_mobius_cycles
        self.use_adaptive_evolution_rate = use_adaptive_evolution_rate
        self.use_multiscale_evolution = use_multiscale_evolution
        
        # 创建多层循环单元
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            # 第一层的输入维度是input_dim，后续层的输入维度是前一层输出
            layer_input_dim = input_dim if i == 0 else (
                self.hidden_dim * 2 if bidirectional else self.hidden_dim
            )
            
            # 前向循环单元
            self.cells.append(
                TwistorTitansCell(
                    input_dim=layer_input_dim,
                    hidden_dim=self.hidden_dim,
                    use_mobius=use_mobius,
                    num_mobius_cycles=num_mobius_cycles,
                    use_adaptive_evolution_rate=use_adaptive_evolution_rate,
                    use_multiscale_evolution=use_multiscale_evolution
                )
            )
            
            # 双向循环：添加反向循环单元
            if bidirectional:
                self.cells.append(
                    TwistorTitansCell(
                        input_dim=layer_input_dim,
                        hidden_dim=self.hidden_dim,
                        use_mobius=use_mobius,
                        num_mobius_cycles=num_mobius_cycles,
                        use_adaptive_evolution_rate=use_adaptive_evolution_rate,
                        use_multiscale_evolution=use_multiscale_evolution
                    )
                )
        
        # Dropout层
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        前向传播：循环处理序列
        
        Args:
            x: 输入序列，形状为 (batch, seq_len, input_dim)
            hidden: 初始隐藏状态，tuple of (omega_h, pi_h) for each layer
                如果为None，初始化为零
        
        Returns:
            (output, final_hidden):
            - output: 输出序列，形状为 (batch, seq_len, hidden_dim) 或 
                     (batch, seq_len, hidden_dim * 2) 如果bidirectional
            - final_hidden: 最终隐藏状态，tuple of (omega_h, pi_h) for each layer
        """
        batch_size, seq_len, _ = x.shape
        
        # 初始化隐藏状态
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device, x.dtype)
        
        # 存储所有时间步的输出
        outputs = []
        
        # 逐时间步处理
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_dim)
            
            # 通过所有层处理
            layer_output = x_t
            new_hidden = []
            
            for layer_idx in range(self.num_layers):
                # 获取当前层的隐藏状态
                if self.bidirectional:
                    # 双向：每层有两个单元（前向和反向）
                    forward_idx = layer_idx * 2
                    backward_idx = layer_idx * 2 + 1
                    
                    # 前向处理
                    forward_hidden = hidden[forward_idx] if hidden else None
                    forward_output, forward_new_hidden = self.cells[forward_idx](
                        layer_output, forward_hidden
                    )
                    new_hidden.append(forward_new_hidden)
                    
                    # 反向处理（从序列末尾开始）
                    backward_t = seq_len - 1 - t
                    backward_input = x[:, backward_t, :]
                    backward_hidden = hidden[backward_idx] if hidden else None
                    backward_output, backward_new_hidden = self.cells[backward_idx](
                        backward_input, backward_hidden
                    )
                    new_hidden.append(backward_new_hidden)
                    
                    # 组合前向和反向输出
                    layer_output = torch.cat([forward_output, backward_output], dim=-1)
                else:
                    # 单向：每层一个单元
                    cell_idx = layer_idx
                    cell_hidden = hidden[cell_idx] if hidden else None
                    layer_output, cell_new_hidden = self.cells[cell_idx](
                        layer_output, cell_hidden
                    )
                    new_hidden.append(cell_new_hidden)
                
                # 应用Dropout（除了最后一层）
                if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                    layer_output = self.dropout_layer(layer_output)
            
            outputs.append(layer_output)
            hidden = tuple(new_hidden)
        
        # 堆叠所有时间步的输出
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_dim or hidden_dim*2)
        
        return output, hidden
    
    def _init_hidden(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple:
        """
        初始化隐藏状态
        
        Returns:
            tuple of (omega_h, pi_h) for each layer
        """
        hidden = []
        num_cells = self.num_layers * (2 if self.bidirectional else 1)
        
        for _ in range(num_cells):
            omega_h = torch.zeros(batch_size, self.hidden_dim // 2, device=device, dtype=dtype)
            pi_h = torch.zeros(batch_size, self.hidden_dim // 2, device=device, dtype=dtype)
            hidden.append((omega_h, pi_h))
        
        return tuple(hidden)
    
    def extra_repr(self) -> str:
        return (f'input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, '
                f'num_layers={self.num_layers}, bidirectional={self.bidirectional}, '
                f'dropout={self.dropout}')

