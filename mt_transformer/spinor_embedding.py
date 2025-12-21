"""
旋量嵌入层 (Spinor Embedding)
将Token映射为复数扭量表示
"""

import torch
import torch.nn as nn
import math


class SpinorEmbedding(nn.Module):
    """
    旋量嵌入层：将Token ID映射为复数扭量表示
    
    扭量表示包含两个分量：
    - ω: 语义内容（semantic content）
    - π: 动量/位置趋势（momentum/position trend）
    
    这两个分量在数学上通过关联方程耦合在一起，比RoPE更本质。
    
    Args:
        vocab_size: 词汇表大小
        dim: 嵌入维度（每个分量的维度，总输出维度为dim*2）
        omega_embedding: 是否使用独立的ω嵌入层（默认True）
        pi_embedding: 是否使用独立的π嵌入层（默认True）
        max_seq_len: 最大序列长度，用于位置编码（默认512）
        use_complex: 是否使用PyTorch原生复数类型（默认False，使用分离实部虚部）
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        omega_embedding: bool = True,
        pi_embedding: bool = True,
        max_seq_len: int = 512,
        use_complex: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.omega_embedding = omega_embedding
        self.pi_embedding = pi_embedding
        self.max_seq_len = max_seq_len
        self.use_complex = use_complex
        
        # 创建嵌入层
        if omega_embedding:
            # ω嵌入层：用于语义内容
            self.omega_emb = nn.Embedding(vocab_size, dim * 2 if not use_complex else dim)
        else:
            self.omega_emb = None
            
        if pi_embedding:
            # π嵌入层：用于动量/位置趋势
            self.pi_emb = nn.Embedding(vocab_size, dim * 2 if not use_complex else dim)
        else:
            self.pi_emb = None
        
        # 如果只使用一个嵌入层，创建共享嵌入
        if not omega_embedding and not pi_embedding:
            raise ValueError("至少需要启用omega_embedding或pi_embedding中的一个")
        elif not omega_embedding:
            # 只使用π嵌入，ω从π派生
            self.omega_emb = None
        elif not pi_embedding:
            # 只使用ω嵌入，π从ω派生
            self.pi_emb = None
        else:
            # 两个都使用，但可以共享权重
            pass
        
        # 位置编码（可选，用于增强位置信息）
        # 注意：twistor 是 omega 和 pi 的拼接，所以总维度是 dim*4
        # 但位置编码只需要 dim*2（因为 omega 和 pi 各占 dim*2）
        # 我们会对 omega 和 pi 分别应用相同的位置编码
        self.register_buffer(
            'pos_encoding',
            self._create_positional_encoding(max_seq_len, dim)
        )
        
    def _create_positional_encoding(self, max_len: int, dim: int) -> torch.Tensor:
        """
        创建位置编码（类似RoPE，但基于扭量理论）
        
        返回形状: (max_len, dim*2) 或 (max_len, dim) 如果use_complex=True
        """
        if self.use_complex:
            # 复数位置编码
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                                (-math.log(10000.0) / dim))
            pos_encoding = torch.zeros(max_len, dim, dtype=torch.complex64)
            pos_encoding[:, 0::2] = torch.sin(position * div_term)
            pos_encoding[:, 1::2] = torch.cos(position * div_term)
            return pos_encoding
        else:
            # 分离实部虚部的位置编码
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            # div_term 的长度是 dim//2，用于生成 sin/cos 对
            div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                                (-math.log(10000.0) / dim))
            pos_encoding = torch.zeros(max_len, dim * 2, dtype=torch.float32)
            
            # 计算位置编码值，形状为 (max_len, dim//2)
            pe_sin = torch.sin(position * div_term)  # (max_len, dim//2)
            pe_cos = torch.cos(position * div_term)  # (max_len, dim//2)
            
            # 填充实部部分（前 dim 个位置）：sin, cos, sin, cos, ...
            # 将 (max_len, dim//2) 扩展为 (max_len, dim) 通过重复
            pe_real = torch.zeros(max_len, dim, dtype=torch.float32)
            pe_real[:, 0::2] = pe_sin  # 偶数位置：sin
            pe_real[:, 1::2] = pe_cos  # 奇数位置：cos
            pos_encoding[:, :dim] = pe_real
            
            # 填充虚部部分（后 dim 个位置）：cos, -sin, cos, -sin, ...
            pe_imag = torch.zeros(max_len, dim, dtype=torch.float32)
            pe_imag[:, 0::2] = pe_cos   # 偶数位置：cos
            pe_imag[:, 1::2] = -pe_sin  # 奇数位置：-sin
            pos_encoding[:, dim:] = pe_imag
            
            return pos_encoding
    
    def forward(
        self,
        token_ids: torch.Tensor,
        use_pos_encoding: bool = True
    ) -> torch.Tensor:
        """
        前向传播：将Token ID映射为扭量表示
        
        Args:
            token_ids: Token ID张量，形状为 (batch_size, seq_len)
            use_pos_encoding: 是否使用位置编码（默认True）
        
        Returns:
            扭量表示，形状为 (batch_size, seq_len, dim*2) 或 (batch_size, seq_len, dim) 如果use_complex=True
        """
        batch_size, seq_len = token_ids.shape
        
        # 获取嵌入
        if self.omega_emb is not None and self.pi_emb is not None:
            # 两个独立的嵌入层
            omega = self.omega_emb(token_ids)  # (batch_size, seq_len, dim*2 or dim)
            pi = self.pi_emb(token_ids)  # (batch_size, seq_len, dim*2 or dim)
        elif self.omega_emb is not None:
            # 只有ω嵌入，π从ω派生（通过关联方程）
            omega = self.omega_emb(token_ids)
            # 简化：π = -conj(omega) 的某种变换
            if self.use_complex:
                pi = -torch.conj(omega)
            else:
                # 分离实部虚部的情况
                omega_dim = omega.shape[-1]
                if omega_dim % 2 == 0:
                    omega_real, omega_imag = omega.chunk(2, dim=-1)
                    pi = torch.cat([-omega_real, omega_imag], dim=-1)
                else:
                    pi = -omega
        else:
            # 只有π嵌入，ω从π派生
            pi = self.pi_emb(token_ids)
            if self.use_complex:
                omega = torch.conj(pi)
            else:
                pi_dim = pi.shape[-1]
                if pi_dim % 2 == 0:
                    pi_real, pi_imag = pi.chunk(2, dim=-1)
                    omega = torch.cat([pi_real, -pi_imag], dim=-1)
                else:
                    omega = pi
        
        # 添加位置编码（在拼接之前，分别应用到 omega 和 pi）
        if use_pos_encoding and seq_len <= self.max_seq_len:
            pos_enc = self.pos_encoding[:seq_len]  # (seq_len, dim*2 or dim)
            pos_enc = pos_enc.unsqueeze(0)  # (1, seq_len, dim*2 or dim)
            # 分别应用到 omega 和 pi
            omega = omega + pos_enc
            pi = pi + pos_enc
        
        # 组合ω和π
        if self.use_complex:
            # 复数情况：直接拼接
            twistor = torch.cat([omega, pi], dim=-1)
        else:
            # 分离实部虚部：拼接
            twistor = torch.cat([omega, pi], dim=-1)
        
        return twistor
    
    def extra_repr(self) -> str:
        return (f'vocab_size={self.vocab_size}, dim={self.dim}, '
                f'omega_embedding={self.omega_embedding}, pi_embedding={self.pi_embedding}, '
                f'use_complex={self.use_complex}')

