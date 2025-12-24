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
        use_multiscale_evolution: bool = True,
        chunk_size: int = 512,
        use_chunk_parallel: bool = True
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
        self.chunk_size = chunk_size
        self.use_chunk_parallel = use_chunk_parallel
        
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
        前向传播：循环处理序列（支持分块并行）
        
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
        
        # 如果启用分块并行且序列长度大于chunk_size，使用分块并行
        if self.use_chunk_parallel and seq_len > self.chunk_size:
            return self._forward_chunk_parallel(x, hidden)
        else:
            # 使用原有的顺序处理方式
            return self._forward_sequential(x, hidden)
    
    def _forward_sequential(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """原有的顺序处理方式（向后兼容）"""
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
    
    def _forward_chunk_parallel(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """分块并行处理方式"""
        batch_size, seq_len, _ = x.shape
        
        # 计算块数
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        # 将序列分成块
        chunks = []
        chunk_outputs = []
        chunk_final_hiddens = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, seq_len)
            chunk = x[:, start_idx:end_idx, :]  # (batch, chunk_len, input_dim)
            chunks.append(chunk)
        
        # 并行处理每个块（每个块独立处理，无依赖）
        for chunk_idx, chunk in enumerate(chunks):
            # 每个块使用独立的隐藏状态（初始化为零）
            chunk_hidden = self._init_hidden(batch_size, x.device, x.dtype)
            
            # 对当前块进行顺序处理
            chunk_output, chunk_final_hidden = self._forward_sequential(chunk, chunk_hidden)
            chunk_outputs.append(chunk_output)
            chunk_final_hiddens.append(chunk_final_hidden)
        
        # 拼接所有块的输出
        output = torch.cat(chunk_outputs, dim=1)  # (batch, seq_len, hidden_dim or hidden_dim*2)
        
        # 块间莫比乌斯变换连接（连接相邻块）
        if num_chunks > 1:
            output = self._connect_chunks_with_mobius(output, num_chunks)
        
        # 使用最后一个块的隐藏状态作为最终隐藏状态
        final_hidden = chunk_final_hiddens[-1]
        
        return output, final_hidden
    
    def _connect_chunks_with_mobius(
        self,
        output: torch.Tensor,
        num_chunks: int
    ) -> torch.Tensor:
        """使用莫比乌斯变换连接块"""
        batch_size, seq_len, dim = output.shape
        chunk_len = self.chunk_size
        
        # 分离ω和π分量
        omega, pi = output.chunk(2, dim=-1)  # 各为 (batch, seq_len, dim//2)
        
        # 对每个块边界进行莫比乌斯变换连接
        for chunk_idx in range(num_chunks - 1):
            # 当前块的结束位置
            end_idx = min((chunk_idx + 1) * chunk_len, seq_len)
            # 下一个块的开始位置
            next_start_idx = end_idx
            
            if next_start_idx >= seq_len:
                break
            
            # 获取边界处的值
            current_end_omega = omega[:, end_idx - 1:end_idx, :]  # (batch, 1, dim//2)
            current_end_pi = pi[:, end_idx - 1:end_idx, :]
            next_start_omega = omega[:, next_start_idx:next_start_idx + 1, :]
            next_start_pi = pi[:, next_start_idx:next_start_idx + 1, :]
            
            # 莫比乌斯变换：交换并取共轭（简化版本）
            # omega_new = pi, pi_new = -omega
            # 在边界处混合
            omega_dim = current_end_omega.shape[-1]
            if omega_dim % 2 == 0:
                # 分离实部虚部
                current_end_omega_real, current_end_omega_imag = current_end_omega.chunk(2, dim=-1)
                current_end_pi_real, current_end_pi_imag = current_end_pi.chunk(2, dim=-1)
                next_start_omega_real, next_start_omega_imag = next_start_omega.chunk(2, dim=-1)
                next_start_pi_real, next_start_pi_imag = next_start_pi.chunk(2, dim=-1)
                
                # 莫比乌斯变换：omega_new = pi, pi_new = -omega
                mobius_omega_real = current_end_pi_real
                mobius_omega_imag = -current_end_pi_imag
                mobius_pi_real = -current_end_omega_real
                mobius_pi_imag = current_end_omega_imag
                
                mobius_omega = torch.cat([mobius_omega_real, mobius_omega_imag], dim=-1)
                mobius_pi = torch.cat([mobius_pi_real, mobius_pi_imag], dim=-1)
            else:
                # 简化处理
                mobius_omega = current_end_pi
                mobius_pi = -current_end_omega
            
            # 混合原始值和莫比乌斯变换值（耦合系数0.1）
            coupling_coeff = 0.1
            omega[:, next_start_idx:next_start_idx + 1, :] = (
                (1 - coupling_coeff) * next_start_omega + coupling_coeff * mobius_omega
            )
            pi[:, next_start_idx:next_start_idx + 1, :] = (
                (1 - coupling_coeff) * next_start_pi + coupling_coeff * mobius_pi
            )
        
        # 重新组合
        output = torch.cat([omega, pi], dim=-1)
        
        return output
    
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
                f'dropout={self.dropout}, chunk_size={self.chunk_size}, '
                f'use_chunk_parallel={self.use_chunk_parallel}')

