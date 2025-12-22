"""
快速测试Recommended配置
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mt_transformer import TwistorHopeArchitecture
from mt_transformer.model_configs import create_full_config

print("=" * 80)
print("测试Recommended配置")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}\n")

# 获取recommended配置
config = create_full_config('recommended', vocab_size=1000)
print("Recommended配置:")
for key, value in config.items():
    print(f"  {key}: {value}")
print()

# 创建模型
print("创建模型...")
try:
    model = TwistorHopeArchitecture(**config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型创建成功")
    print(f"✓ 参数量: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # 测试前向传播
    print("\n测试前向传播...")
    batch_size = 2
    seq_len = 16
    token_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    
    output, constraint_loss = model(token_ids, return_constraint_loss=True)
    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ 约束损失: {constraint_loss.item():.6f}")
    
    # 显存使用
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        print(f"\nGPU显存:")
        print(f"  已分配: {allocated:.1f} MB")
        print(f"  已缓存: {reserved:.1f} MB")
        print(f"  总显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        
        # 估算完整训练所需显存
        estimated_mem = allocated * 3  # 粗略估算（模型+激活+梯度）
        print(f"\n预估训练显存: ~{estimated_mem:.1f} MB ({estimated_mem/1024:.2f} GB)")
        
        if estimated_mem / 1024 > 3.5:
            print("⚠️  警告: Recommended配置可能需要较大显存，建议减小batch size")
        else:
            print("✓ 显存需求在安全范围内")
    
    print("\n" + "=" * 80)
    print("✓ Recommended配置测试通过！")
    print("\n运行完整训练:")
    print("  python examples/train_gpu.py recommended")
    print("=" * 80)
    
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()

