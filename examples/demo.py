"""
MT-Transformer核心组件演示和测试
展示如何使用各个组件以及HOPE架构集成
"""

import torch
import torch.nn as nn
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mt_transformer import (
    SpinorEmbedding,
    IncidenceAttention,
    MobiusLayer,
    NestedOptimizer,
    HopeIntegration
)


def test_spinor_embedding():
    """测试旋量嵌入层"""
    print("=" * 60)
    print("测试旋量嵌入层 (SpinorEmbedding)")
    print("=" * 60)
    
    vocab_size = 1000
    dim = 128
    batch_size = 2
    seq_len = 10
    
    # 创建嵌入层
    # 注意：SpinorEmbedding的dim参数是每个分量的维度
    # 输出维度是dim*2（ω和π各占dim）
    embedding = SpinorEmbedding(
        vocab_size=vocab_size,
        dim=dim // 2,  # 每个分量的维度，总输出是dim
        use_complex=False
    )
    
    # 创建随机Token IDs
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入Token IDs形状: {token_ids.shape}")
    
    # 前向传播
    output = embedding(token_ids)
    print(f"输出扭量表示形状: {output.shape}")
    print(f"预期形状: ({batch_size}, {seq_len}, {dim})")
    assert output.shape == (batch_size, seq_len, dim), "形状不匹配！"
    
    print("✓ 旋量嵌入层测试通过\n")


def test_mobius_layer():
    """测试莫比乌斯层"""
    print("=" * 60)
    print("测试莫比乌斯层 (MobiusLayer)")
    print("=" * 60)
    
    dim = 128
    batch_size = 2
    seq_len = 10
    
    # 创建莫比乌斯层
    mobius = MobiusLayer(dim=dim, coupling_coeff=0.1, use_complex=False)
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, dim)
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    output = mobius(x)
    print(f"输出形状: {output.shape}")
    assert output.shape == x.shape, "形状不匹配！"
    
    # 测试莫比乌斯变换的效果
    print(f"输入均值: {x.mean().item():.4f}")
    print(f"输出均值: {output.mean().item():.4f}")
    print(f"输出与输入的差异: {(output - x).abs().mean().item():.4f}")
    
    print("✓ 莫比乌斯层测试通过\n")


def test_incidence_attention():
    """测试关联注意力机制"""
    print("=" * 60)
    print("测试关联注意力机制 (IncidenceAttention)")
    print("=" * 60)
    
    dim = 128
    num_heads = 8
    batch_size = 2
    seq_len = 10
    
    # 创建关联注意力层
    # 注意：输入应该是扭量表示，维度为dim*2（如果嵌入输出是dim）
    # 但这里我们测试的是dim，所以需要确保dim是偶数
    attention_dim = dim if dim % 2 == 0 else dim + 1
    attention = IncidenceAttention(
        dim=attention_dim,
        num_heads=num_heads,
        dropout=0.1,
        use_mobius_term=True,
        use_complex=False
    )
    
    # 创建随机输入（扭量表示，维度为attention_dim）
    x = torch.randn(batch_size, seq_len, attention_dim)
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    output = attention(x)
    print(f"输出形状: {output.shape}")
    assert output.shape == x.shape, "形状不匹配！"
    
    # 测试返回注意力权重
    output_with_weights, weights = attention(x, return_attention_weights=True)
    print(f"注意力权重形状: {weights.shape}")
    print(f"注意力权重和: {weights.sum(dim=-1).mean().item():.4f} (应该接近1.0)")
    
    print("✓ 关联注意力机制测试通过\n")


def test_nested_optimizer():
    """测试嵌套优化器"""
    print("=" * 60)
    print("测试嵌套优化器 (NestedOptimizer)")
    print("=" * 60)
    
    dim = 128
    batch_size = 2
    seq_len = 10
    
    # 创建一个简单的模块
    module = nn.Sequential(
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Linear(dim, dim)
    )
    
    # 包装为嵌套优化器
    nested = NestedOptimizer(
        module=module,
        inner_optimizer=torch.optim.Adam,
        inner_lr=1e-4
    )
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, dim)
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    output = nested(x, update_context=True)
    print(f"输出形状: {output.shape}")
    
    print("✓ 嵌套优化器测试通过\n")


def test_hope_integration():
    """测试HOPE架构集成"""
    print("=" * 60)
    print("测试HOPE架构集成 (HopeIntegration)")
    print("=" * 60)
    
    vocab_size = 1000
    dim = 128
    batch_size = 2
    seq_len = 10
    
    # 创建HOPE集成模型
    model = HopeIntegration(
        vocab_size=vocab_size,
        dim=dim,
        num_titan_layers=2,
        use_nested_optimizer=False
    )
    
    # 创建随机Token IDs
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入Token IDs形状: {token_ids.shape}")
    
    # 前向传播
    output = model(token_ids)
    print(f"输出形状: {output.shape}")
    print(f"预期形状: ({batch_size}, {seq_len}, {dim})")
    assert output.shape == (batch_size, seq_len, dim), "形状不匹配！"
    
    # 测试梯度
    loss = output.mean()
    loss.backward()
    print("✓ 梯度计算正常")
    
    print("✓ HOPE架构集成测试通过\n")


def test_complete_pipeline():
    """测试完整流程"""
    print("=" * 60)
    print("测试完整流程：从Token到输出")
    print("=" * 60)
    
    vocab_size = 1000
    dim = 128
    batch_size = 2
    seq_len = 10
    
    # 创建各个组件
    # SpinorEmbedding的dim参数是每个分量的维度，输出是dim*2
    # 所以如果我们要得到总维度为dim的输出，应该设置dim=dim//2
    embedding = SpinorEmbedding(vocab_size=vocab_size, dim=dim // 2)
    # 嵌入输出维度是dim（因为dim//2 * 2 = dim）
    embedding_output_dim = dim
    # 确保是偶数（因为包含ω和π两个分量）
    if embedding_output_dim % 2 != 0:
        embedding_output_dim += 1
    
    attention = IncidenceAttention(dim=embedding_output_dim)
    mobius = MobiusLayer(dim=embedding_output_dim)
    
    # 创建Token IDs
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"1. Token IDs: {token_ids.shape}")
    
    # 旋量嵌入
    x = embedding(token_ids)
    print(f"2. 旋量嵌入后: {x.shape}")
    
    # 如果维度不匹配，使用线性投影调整
    if x.shape[-1] != embedding_output_dim:
        proj = nn.Linear(x.shape[-1], embedding_output_dim)
        x = proj(x)
        print(f"2.5. 维度调整后: {x.shape}")
    
    # 关联注意力
    x = attention(x)
    print(f"3. 关联注意力后: {x.shape}")
    
    # 莫比乌斯层
    x = mobius(x)
    print(f"4. 莫比乌斯层后: {x.shape}")
    
    print("✓ 完整流程测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("MT-Transformer 核心组件测试")
    print("=" * 60 + "\n")
    
    try:
        test_spinor_embedding()
        test_mobius_layer()
        test_incidence_attention()
        test_nested_optimizer()
        test_hope_integration()
        test_complete_pipeline()
        
        print("=" * 60)
        print("所有测试通过！✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

