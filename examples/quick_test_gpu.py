"""
快速GPU测试脚本 - 运行少量epoch查看训练效果
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mt_transformer import TwistorHopeArchitecture

print("=" * 80)
print("快速GPU训练测试")
print("=" * 80)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n")

# 创建小模型（快速测试）
print("创建模型...")
model = TwistorHopeArchitecture(
    vocab_size=1000,
    dim=64,  # 减小维度加速
    hidden_dim=64,
    num_recurrent_layers=1,  # 减少层数
    num_memories=2,
    use_nested_learning=True,
    bidirectional=False,
    dropout=0.1
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"✓ 模型参数量: {num_params:,} ({num_params/1e6:.2f}M)")

# 创建优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 创建小数据集
batch_size = 4
seq_len = 16
num_batches = 5  # 只测试5个batch

print(f"\n开始训练测试 ({num_batches} 个batch)...")
print("-" * 80)

model.train()
total_loss = 0.0

for i in range(num_batches):
    # 创建随机数据
    token_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    
    # 前向传播
    optimizer.zero_grad()
    output, constraint_loss = model(token_ids, return_constraint_loss=True)
    
    # 计算损失
    target_emb = model.embedding(targets)
    target_proj = target_emb[..., :output.shape[-1]]
    loss = nn.functional.mse_loss(output, target_proj) + constraint_loss
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()
    print(f"Batch [{i+1}/{num_batches}]: Loss = {loss.item():.6f}, Constraint = {constraint_loss.item():.6f}")

avg_loss = total_loss / num_batches
print("-" * 80)
print(f"平均损失: {avg_loss:.6f}")

# 显存使用
if device.type == 'cuda':
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
    print(f"GPU显存: 已分配 {allocated:.1f} MB, 已缓存 {reserved:.1f} MB")

print("\n✓ 快速测试完成！训练脚本可以正常运行。")
print("\n完整训练建议:")
print("  - 运行: python examples/train_gpu.py")
print("  - 默认配置: 10 epochs, 1000样本")
print("  - 预计时间: 5-10分钟（取决于GPU）")
print("=" * 80)

