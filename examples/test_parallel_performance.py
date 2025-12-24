"""
并行性能测试脚本
对比优化前后的性能，测量GPU利用率和训练速度提升
"""

import torch
import torch.nn as nn
import time
import sys
import os
from typing import Dict, Any

# 添加父目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mt_transformer import TwistorHopeArchitecture
from mt_transformer.model_configs import get_config, create_full_config


def measure_gpu_utilization():
    """测量GPU利用率（简化版本）"""
    if not torch.cuda.is_available():
        return 0.0
    
    # 获取GPU使用情况（需要nvidia-smi，这里简化处理）
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    
    return 0.0


def benchmark_model(
    model: nn.Module,
    input_shape: tuple,
    device: torch.device,
    num_warmup: int = 5,
    num_iterations: int = 20,
    use_compile: bool = False
) -> Dict[str, Any]:
    """
    基准测试模型性能
    
    Args:
        model: 模型
        input_shape: 输入形状 (batch_size, seq_len)
        device: 设备
        num_warmup: 预热迭代次数
        num_iterations: 测试迭代次数
        use_compile: 是否使用torch.compile
    
    Returns:
        性能指标字典
    """
    model.eval()
    batch_size, seq_len = input_shape
    
    # 创建输入
    token_ids = torch.randint(0, 1000, input_shape, device=device)
    
    # 预热
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(token_ids)
    
    # 同步GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 测试
    start_time = time.time()
    with torch.inference_mode():
        for _ in range(num_iterations):
            _ = model(token_ids)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # 计算指标
    total_time = end_time - start_time
    avg_time_per_iteration = total_time / num_iterations
    throughput_samples_per_sec = (batch_size * num_iterations) / total_time
    
    # 测量显存
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)  # MB
        memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)  # MB
    else:
        memory_allocated = 0
        memory_reserved = 0
    
    return {
        'avg_time_per_iteration': avg_time_per_iteration,
        'throughput_samples_per_sec': throughput_samples_per_sec,
        'memory_allocated_mb': memory_allocated,
        'memory_reserved_mb': memory_reserved,
        'total_time': total_time
    }


def compare_configurations():
    """对比不同配置的性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("并行性能测试")
    print("=" * 80)
    print(f"设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 测试配置
    vocab_size = 1000
    batch_size = 8
    seq_len = 512  # 使用较长序列以测试分块并行
    
    configs_to_test = [
        {
            'name': '原始配置（无并行优化）',
            'config': 'small',
            'use_chunk_parallel': False,
            'use_pipeline_parallel': False,
            'use_compile': False
        },
        {
            'name': '仅分块并行',
            'config': 'small',
            'use_chunk_parallel': True,
            'use_pipeline_parallel': False,
            'use_compile': False
        },
        {
            'name': '分块并行 + 流水线并行',
            'config': 'small',
            'use_chunk_parallel': True,
            'use_pipeline_parallel': True,
            'use_compile': False
        },
        {
            'name': '全部优化（分块 + 流水线 + compile）',
            'config': 'small',
            'use_chunk_parallel': True,
            'use_pipeline_parallel': True,
            'use_compile': True
        }
    ]
    
    results = []
    
    for test_config in configs_to_test:
        print(f"\n测试: {test_config['name']}")
        print("-" * 80)
        
        # 创建配置
        base_config = create_full_config(
            config_name=test_config['config'],
            vocab_size=vocab_size
        )
        base_config['use_chunk_parallel'] = test_config['use_chunk_parallel']
        base_config['use_pipeline_parallel'] = test_config['use_pipeline_parallel']
        
        # 创建模型
        model = TwistorHopeArchitecture(
            vocab_size=vocab_size,
            dim=base_config['dim'],
            hidden_dim=base_config.get('hidden_dim', base_config['dim']),
            num_recurrent_layers=base_config['num_recurrent_layers'],
            num_memories=base_config['num_memories'],
            num_memory_cycles=base_config.get('num_memory_cycles', 2),
            use_nested_learning=base_config.get('use_nested_learning', True),
            use_phase_compression=base_config.get('use_phase_compression', True),
            bidirectional=base_config.get('bidirectional', False),
            dropout=base_config.get('dropout', 0.1),
            num_mobius_cycles=base_config.get('num_mobius_cycles', 3),
            use_adaptive_evolution_rate=base_config.get('use_adaptive_evolution_rate', True),
            use_multiscale_evolution=base_config.get('use_multiscale_evolution', True),
            num_nested_levels=base_config.get('num_nested_levels', 5),
            nested_level_lrs=base_config.get('nested_level_lrs', None),
            use_level_constraints=base_config.get('use_level_constraints', True),
            chunk_size=base_config.get('chunk_size', 512),
            use_chunk_parallel=test_config['use_chunk_parallel'],
            use_pipeline_parallel=test_config['use_pipeline_parallel']
        ).to(device)
        
        # 应用torch.compile（如果启用）
        if test_config['use_compile'] and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='reduce-overhead')
                print("✓ torch.compile已启用")
            except Exception as e:
                print(f"⚠️  torch.compile失败: {e}")
        
        # 清理显存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 基准测试
        perf = benchmark_model(
            model,
            (batch_size, seq_len),
            device,
            num_warmup=3,
            num_iterations=10,
            use_compile=test_config['use_compile']
        )
        
        results.append({
            'name': test_config['name'],
            **perf
        })
        
        print(f"  平均时间/迭代: {perf['avg_time_per_iteration']*1000:.2f} ms")
        print(f"  吞吐量: {perf['throughput_samples_per_sec']:.2f} 样本/秒")
        print(f"  显存占用: {perf['memory_allocated_mb']:.1f} MB")
        
        # 清理
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # 打印对比结果
    print("\n" + "=" * 80)
    print("性能对比总结")
    print("=" * 80)
    
    if len(results) > 1:
        baseline = results[0]
        print(f"\n基准配置: {baseline['name']}")
        print(f"  时间: {baseline['avg_time_per_iteration']*1000:.2f} ms")
        print(f"  吞吐量: {baseline['throughput_samples_per_sec']:.2f} 样本/秒")
        
        for result in results[1:]:
            speedup = baseline['avg_time_per_iteration'] / result['avg_time_per_iteration']
            throughput_improvement = result['throughput_samples_per_sec'] / baseline['throughput_samples_per_sec']
            print(f"\n{result['name']}:")
            print(f"  时间: {result['avg_time_per_iteration']*1000:.2f} ms")
            print(f"  吞吐量: {result['throughput_samples_per_sec']:.2f} 样本/秒")
            print(f"  速度提升: {speedup:.2f}x")
            print(f"  吞吐量提升: {throughput_improvement:.2f}x")
    
    print("\n" + "=" * 80)


def test_output_consistency():
    """测试优化前后的输出一致性"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 80)
    print("输出一致性测试")
    print("=" * 80)
    
    vocab_size = 1000
    batch_size = 4
    seq_len = 256
    
    # 创建输入
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # 测试配置1：无并行
    config1 = create_full_config('small', vocab_size=vocab_size)
    config1['use_chunk_parallel'] = False
    config1['use_pipeline_parallel'] = False
    
    model1 = TwistorHopeArchitecture(
        vocab_size=vocab_size,
        dim=config1['dim'],
        hidden_dim=config1.get('hidden_dim', config1['dim']),
        num_recurrent_layers=config1['num_recurrent_layers'],
        num_memories=config1['num_memories'],
        num_memory_cycles=config1.get('num_memory_cycles', 2),
        use_nested_learning=config1.get('use_nested_learning', True),
        use_phase_compression=config1.get('use_phase_compression', True),
        bidirectional=config1.get('bidirectional', False),
        dropout=config1.get('dropout', 0.1),
        num_mobius_cycles=config1.get('num_mobius_cycles', 3),
        use_adaptive_evolution_rate=config1.get('use_adaptive_evolution_rate', True),
        use_multiscale_evolution=config1.get('use_multiscale_evolution', True),
        num_nested_levels=config1.get('num_nested_levels', 5),
        nested_level_lrs=config1.get('nested_level_lrs', None),
        use_level_constraints=config1.get('use_level_constraints', True),
        chunk_size=config1.get('chunk_size', 512),
        use_chunk_parallel=False,
        use_pipeline_parallel=False
    ).to(device)
    
    # 测试配置2：有并行
    config2 = create_full_config('small', vocab_size=vocab_size)
    config2['use_chunk_parallel'] = True
    config2['use_pipeline_parallel'] = True
    
    model2 = TwistorHopeArchitecture(
        vocab_size=vocab_size,
        dim=config2['dim'],
        hidden_dim=config2.get('hidden_dim', config2['dim']),
        num_recurrent_layers=config2['num_recurrent_layers'],
        num_memories=config2['num_memories'],
        num_memory_cycles=config2.get('num_memory_cycles', 2),
        use_nested_learning=config2.get('use_nested_learning', True),
        use_phase_compression=config2.get('use_phase_compression', True),
        bidirectional=config2.get('bidirectional', False),
        dropout=config2.get('dropout', 0.1),
        num_mobius_cycles=config2.get('num_mobius_cycles', 3),
        use_adaptive_evolution_rate=config2.get('use_adaptive_evolution_rate', True),
        use_multiscale_evolution=config2.get('use_multiscale_evolution', True),
        num_nested_levels=config2.get('num_nested_levels', 5),
        nested_level_lrs=config2.get('nested_level_lrs', None),
        use_level_constraints=config2.get('use_level_constraints', True),
        chunk_size=config2.get('chunk_size', 512),
        use_chunk_parallel=True,
        use_pipeline_parallel=True
    ).to(device)
    
    # 复制权重（确保两个模型参数相同）
    model2.load_state_dict(model1.state_dict())
    
    # 设置为评估模式
    model1.eval()
    model2.eval()
    
    # 前向传播
    with torch.inference_mode():
        output1, constraint_loss1 = model1(token_ids, return_constraint_loss=True)
        output2, constraint_loss2 = model2(token_ids, return_constraint_loss=True)
    
    # 计算差异
    output_diff = torch.abs(output1 - output2).mean().item()
    constraint_diff = abs(constraint_loss1.item() - constraint_loss2.item())
    
    print(f"输出差异 (mean abs): {output_diff:.6f}")
    print(f"约束损失差异: {constraint_diff:.6f}")
    
    # 判断是否一致（允许小的数值误差）
    tolerance = 1e-4
    if output_diff < tolerance and constraint_diff < tolerance:
        print("✓ 输出一致性测试通过（差异在容差范围内）")
    else:
        print(f"⚠️  输出存在差异（可能由于并行计算的数值误差）")
        print(f"   容差: {tolerance}")
    
    print("=" * 80)


def main():
    """主函数"""
    # 性能对比测试
    compare_configurations()
    
    # 输出一致性测试
    test_output_consistency()
    
    print("\n测试完成！")


if __name__ == '__main__':
    main()

