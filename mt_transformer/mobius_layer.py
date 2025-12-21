"""
莫比乌斯层 (Möbius Layer)
实现拓扑循环结构，支持复数张量的莫比乌斯变换
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MobiusLayer(nn.Module):
    """
    莫比乌斯层：实现拓扑循环结构
    
    基于文档中的设计，执行莫比乌斯变换：
    - 分离旋量分量（ω和π）
    - 执行莫比乌斯变换：交换位置并取共轭
    - 实现残差连接（拓扑耦合系数可配置）
    
    Args:
        dim: 输入维度（应该是偶数，因为包含ω和π两个分量）
        coupling_coeff: 拓扑耦合系数，控制莫比乌斯变换的强度（默认0.1）
        use_complex: 是否使用PyTorch原生复数类型（默认False，使用分离实部虚部）
    """
    
    def __init__(self, dim: int, coupling_coeff: float = 0.1, use_complex: bool = False):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"维度必须是偶数，因为包含ω和π两个分量，当前维度: {dim}")
        
        self.dim = dim
        self.coupling_coeff = coupling_coeff
        self.use_complex = use_complex
        
        # 定义辛形式的扭转矩阵 (Symplectic Twist)
        # 这个矩阵用于旋量分量的交换操作
        self.register_buffer('twist_matrix', torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32))
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播：执行莫比乌斯变换
        
        Args:
            z: 输入张量，形状为 (batch_size, seq_len, dim)
                如果use_complex=False，z是实数张量，前dim//2是ω的实部虚部，后dim//2是π的实部虚部
                如果use_complex=True，z是复数张量，前dim//2是ω，后dim//2是π
        
        Returns:
            变换后的张量，形状与输入相同
        """
        batch_size, seq_len, dim = z.shape
        
        if dim != self.dim:
            raise ValueError(f"输入维度 {dim} 与初始化维度 {self.dim} 不匹配")
        
        # 分离旋量分量（ω和π）
        # 假设z的前半部分是ω，后半部分是π
        omega, pi = z.chunk(2, dim=-1)
        
        if self.use_complex:
            # 使用PyTorch原生复数类型
            # omega和pi都是复数张量
            # 莫比乌斯变换：交换位置并取共轭
            omega_new = torch.conj(pi)
            pi_new = -torch.conj(omega)  # 负号代表自旋的相位翻转
            
            # 重新组合
            z_twisted = torch.cat([omega_new, pi_new], dim=-1)
        else:
            # 使用分离实部虚部的方式
            # omega和pi的形状都是 (batch_size, seq_len, dim//2)
            # 我们需要将它们视为复数，执行共轭操作
            
            # 将实部虚部分离（假设最后维度是2的倍数，前一半是实部，后一半是虚部）
            omega_dim = omega.shape[-1]
            if omega_dim % 2 == 0:
                # 如果维度是偶数，假设前一半是实部，后一半是虚部
                omega_real, omega_imag = omega.chunk(2, dim=-1)
                pi_real, pi_imag = pi.chunk(2, dim=-1)
                
                # 共轭操作：实部不变，虚部取反
                # 莫比乌斯变换：omega_new = conj(pi), pi_new = -conj(omega)
                omega_new_real = pi_real
                omega_new_imag = -pi_imag
                pi_new_real = -omega_real
                pi_new_imag = omega_imag
                
                # 重新组合
                omega_new = torch.cat([omega_new_real, omega_new_imag], dim=-1)
                pi_new = torch.cat([pi_new_real, pi_new_imag], dim=-1)
            else:
                # 如果维度是奇数，我们假设omega和pi本身就是复数分量
                # 这里简化处理：直接交换并取负
                omega_new = pi
                pi_new = -omega
            
            z_twisted = torch.cat([omega_new, pi_new], dim=-1)
        
        # 混合原始信号（残差连接）
        # 拓扑耦合系数控制莫比乌斯变换的强度
        output = z + self.coupling_coeff * z_twisted
        
        return output
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, coupling_coeff={self.coupling_coeff}, use_complex={self.use_complex}'

