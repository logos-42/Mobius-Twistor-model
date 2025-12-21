"""
MT-HOPE架构演示和测试
展示完整的扭量莫比乌斯-HOPE架构的使用
"""

import torch
import torch.nn as nn
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mt_transformer import MTHopeArchitecture


def test_mt_hope_architecture():
    """测试MT-HOPE完整架构"""
    print("=" * 60)
    print("测试MT-HOPE完整架构 (MTHopeArchitecture)")
    print("=" * 60)
    
    vocab_size = 1000
    dim = 128
    batch_size = 2
    seq_len = 10
    
    # 创建MT-HOPE模型
    model = MTHopeArchitecture(
        vocab_size=vocab_size,
        dim=dim,
        num_self_modifying_layers=2,
        num_memories=3,
        num_memory_cycles=2,
        use_nested_learning=True,
        use_phase_compression=True,
        dropout=0.1
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建随机Token IDs
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"\n输入Token IDs形状: {token_ids.shape}")
    
    # 前向传播
    output, constraint_loss = model(token_ids, return_constraint_loss=True)
    print(f"输出形状: {output.shape}")
    print(f"预期形状: ({batch_size}, {seq_len}, {dim})")
    print(f"关联约束损失: {constraint_loss.item():.6f}")
    assert output.shape == (batch_size, seq_len, dim), "形状不匹配！"
    
    # 测试梯度
    loss = output.mean() + constraint_loss
    loss.backward()
    print("✓ 梯度计算正常")
    
    # 获取记忆状态
    memory_state = model.get_memory_state()
    print(f"记忆状态形状: {memory_state.shape}")
    
    # 创建嵌套优化器
    model.create_nested_optimizers()
    print("✓ 嵌套优化器创建成功")
    
    print("\n✓ MT-HOPE架构测试通过\n")


def test_architecture_components():
    """测试架构的各个组件"""
    print("=" * 60)
    print("测试MT-HOPE架构组件")
    print("=" * 60)
    
    from mt_transformer import (
        TwistorSelfModifying,
        TwistorMemorySystem,
        TwistorNestedLearning
    )
    
    dim = 128
    batch_size = 2
    seq_len = 10
    
    # 测试扭量自我修正
    print("\n1. 测试扭量自我修正模块...")
    self_modifying = TwistorSelfModifying(
        dim=dim,
        num_layers=2,
        use_evolution=True,
        use_adaptive_attention=True,
        use_adaptive_mobius=True
    )
    x = torch.randn(batch_size, seq_len, dim)
    output = self_modifying(x)
    print(f"   输入: {x.shape}, 输出: {output.shape}")
    assert output.shape == x.shape, "形状不匹配！"
    print("   ✓ 扭量自我修正模块正常")
    
    # 测试扭量记忆系统
    print("\n2. 测试扭量记忆系统...")
    memory_system = TwistorMemorySystem(
        dim=dim,
        num_memories=3,
        use_phase_compression=True
    )
    x = torch.randn(batch_size, seq_len, dim)
    output = memory_system(x, update_memory=True)
    print(f"   输入: {x.shape}, 输出: {output.shape}")
    assert output.shape == x.shape, "形状不匹配！"
    memory_state = memory_system.get_memory_state()
    print(f"   记忆状态: {memory_state.shape}")
    print("   ✓ 扭量记忆系统正常")
    
    # 测试扭量嵌套学习
    print("\n3. 测试扭量嵌套学习...")
    nested_learning = TwistorNestedLearning(dim=dim)
    x = torch.randn(batch_size, seq_len, dim)
    output, constraint_loss = nested_learning(x)
    print(f"   输入: {x.shape}, 输出: {output.shape}")
    print(f"   约束损失: {constraint_loss.item():.6f}")
    assert output.shape == x.shape, "形状不匹配！"
    nested_learning.create_nested_optimizers()
    print("   ✓ 扭量嵌套学习正常")
    
    print("\n✓ 所有组件测试通过\n")


def test_complete_pipeline():
    """测试完整流程"""
    print("=" * 60)
    print("测试完整流程：从Token到输出")
    print("=" * 60)
    
    vocab_size = 1000
    dim = 128
    batch_size = 2
    seq_len = 10
    
    model = MTHopeArchitecture(
        vocab_size=vocab_size,
        dim=dim,
        num_self_modifying_layers=2,
        use_nested_learning=True
    )
    
    # 创建Token IDs
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"1. Token IDs: {token_ids.shape}")
    
    # 完整前向传播
    output, constraint_loss = model(token_ids, return_constraint_loss=True)
    print(f"2. 最终输出: {output.shape}")
    print(f"3. 关联约束损失: {constraint_loss.item():.6f}")
    
    # 测试多次前向传播（模拟训练）
    print("\n4. 模拟多次前向传播（测试记忆更新）...")
    for i in range(3):
        output, _ = model(token_ids, return_constraint_loss=True)
        memory = model.get_memory_state()
        print(f"   步骤 {i+1}: 输出均值={output.mean().item():.4f}, "
              f"记忆均值={memory.mean().item():.4f}")
    
    print("\n✓ 完整流程测试通过\n")


def test_architecture_variants():
    """测试不同的架构变体"""
    print("=" * 60)
    print("测试架构变体")
    print("=" * 60)
    
    vocab_size = 1000
    dim = 128
    batch_size = 2
    seq_len = 10
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 变体1：不使用嵌套学习
    print("\n1. 测试不使用嵌套学习的变体...")
    model1 = MTHopeArchitecture(
        vocab_size=vocab_size,
        dim=dim,
        use_nested_learning=False
    )
    output1 = model1(token_ids)
    print(f"   输出形状: {output1.shape}")
    print("   ✓ 变体1正常")
    
    # 变体2：不使用相位压缩
    print("\n2. 测试不使用相位压缩的变体...")
    model2 = MTHopeArchitecture(
        vocab_size=vocab_size,
        dim=dim,
        use_phase_compression=False
    )
    output2 = model2(token_ids)
    print(f"   输出形状: {output2.shape}")
    print("   ✓ 变体2正常")
    
    # 变体3：更多自我修正层
    print("\n3. 测试更多自我修正层的变体...")
    model3 = MTHopeArchitecture(
        vocab_size=vocab_size,
        dim=dim,
        num_self_modifying_layers=4
    )
    output3 = model3(token_ids)
    print(f"   输出形状: {output3.shape}")
    print("   ✓ 变体3正常")
    
    print("\n✓ 所有架构变体测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("MT-HOPE架构完整测试")
    print("=" * 60 + "\n")
    
    try:
        test_mt_hope_architecture()
        test_architecture_components()
        test_complete_pipeline()
        test_architecture_variants()
        
        print("=" * 60)
        print("所有测试通过！✓")
        print("=" * 60)
        print("\nMT-HOPE架构已成功实现：")
        print("- 扭量自我修正机制 ✓")
        print("- 扭量记忆系统 ✓")
        print("- 扭量嵌套学习 ✓")
        print("- 完整架构集成 ✓")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

