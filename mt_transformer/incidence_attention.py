"""
关联注意力机制 (Incidence Attention)
基于扭量关联度的注意力计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class IncidenceAttention(nn.Module):
    """
    关联注意力机制：基于扭量关联度的注意力计算
    
    在扭量理论中，两个扭量"相交"意味着它们位于同一条光线上，代表极强的相关性。
    关联度 = 相似度 + 莫比乌斯项（缠绕数）
    
    Args:
        dim: 输入维度（应该是偶数，因为包含ω和π两个分量）
        num_heads: 注意力头数（默认8）
        dropout: Dropout比率（默认0.1）
        use_mobius_term: 是否使用莫比乌斯项（默认True）
        use_complex: 是否使用PyTorch原生复数类型（默认False）
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_mobius_term: bool = True,
        use_complex: bool = False
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"维度必须是偶数，因为包含ω和π两个分量，当前维度: {dim}")
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.use_mobius_term = use_mobius_term
        self.use_complex = use_complex
        
        if self.head_dim * num_heads != dim:
            raise ValueError(f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除")
        
        # 查询、键、值投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # 莫比乌斯项的可学习参数
        if use_mobius_term:
            # 莫比乌斯项用于表示拓扑缠绕关系
            self.mobius_weight = nn.Parameter(torch.randn(1) * 0.01)
            # 可学习的缠绕矩阵
            self.register_buffer('mobius_matrix', torch.eye(dim // 2, dtype=torch.float32))
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def _compute_incidence_metric(
        self,
        q: torch.Tensor,
        k: torch.Tensor
    ) -> torch.Tensor:
        """
        计算扭量关联度（Incidence Metric）
        
        Args:
            q: 查询张量，形状为 (batch_size, num_heads, seq_len_q, head_dim)
            k: 键张量，形状为 (batch_size, num_heads, seq_len_k, head_dim)
        
        Returns:
            关联度矩阵，形状为 (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        # 基础相似度（点积）
        similarity = torch.matmul(q, k.transpose(-2, -1))  # (batch, heads, seq_q, seq_k)
        
        # 莫比乌斯项：表示拓扑缠绕关系
        if self.use_mobius_term:
            # 分离ω和π分量
            # 注意：head_dim可能不是偶数，我们需要在输入维度（dim）层面分离
            # 这里我们假设在多头之前已经分离了ω和π
            # 为了简化，我们在head_dim层面尝试分离，如果不行就使用整个head_dim
            head_dim = q.shape[-1]
            if head_dim % 2 == 0:
                q_omega, q_pi = q.chunk(2, dim=-1)
                k_omega, k_pi = k.chunk(2, dim=-1)
                
                # 计算缠绕数：通过ω和π的交叉关联
                # M_ij = <q_omega_i, k_pi_j> + <q_pi_i, k_omega_j>
                twist_term_1 = torch.matmul(q_omega, k_pi.transpose(-2, -1))
                twist_term_2 = torch.matmul(q_pi, k_omega.transpose(-2, -1))
                mobius_term = twist_term_1 + twist_term_2
            else:
                # 如果head_dim是奇数，使用整个head_dim计算一个简化的缠绕项
                # 使用交叉注意力模式
                mobius_term = torch.matmul(q, k.transpose(-2, -1)) * 0.5
            
            # 应用可学习的权重
            incidence = similarity + self.mobius_weight * mobius_term
        else:
            incidence = similarity
        
        return incidence
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """
        前向传播：计算关联注意力
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, dim)
            mask: 注意力掩码，形状为 (batch_size, seq_len) 或 (batch_size, seq_len, seq_len)
            return_attention_weights: 是否返回注意力权重
        
        Returns:
            输出张量，形状为 (batch_size, seq_len, dim)
            如果return_attention_weights=True，还返回注意力权重
        """
        batch_size, seq_len, dim = x.shape
        
        if dim != self.dim:
            raise ValueError(f"输入维度 {dim} 与初始化维度 {self.dim} 不匹配")
        
        # 投影到查询、键、值
        q = self.q_proj(x)  # (batch, seq, dim)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 现在形状: (batch, num_heads, seq_len, head_dim)
        
        # 计算关联度
        incidence = self._compute_incidence_metric(q, k)
        
        # 应用缩放
        incidence = incidence * self.scale
        
        # 应用掩码
        if mask is not None:
            if mask.dim() == 2:
                # (batch, seq) -> (batch, 1, 1, seq)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # (batch, seq_q, seq_k) -> (batch, 1, seq_q, seq_k)
                mask = mask.unsqueeze(1)
            # 掩码值设为负无穷，softmax后为0
            incidence = incidence.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化
        attention_weights = F.softmax(incidence, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        # 应用注意力权重到值
        output = torch.matmul(attention_weights, v)  # (batch, heads, seq, head_dim)
        
        # 重塑回原始形状
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, dim
        )
        
        # 输出投影
        output = self.out_proj(output)
        
        if return_attention_weights:
            # 返回平均注意力权重（跨头平均）
            attention_weights_avg = attention_weights.mean(dim=1)  # (batch, seq_q, seq_k)
            return output, attention_weights_avg
        
        return output
    
    def extra_repr(self) -> str:
        return (f'dim={self.dim}, num_heads={self.num_heads}, '
                f'use_mobius_term={self.use_mobius_term}, use_complex={self.use_complex}')

