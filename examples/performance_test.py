"""
扭量化HOPE架构性能测试脚本
测试模型参数量、训练速度、内存占用等性能指标
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
from mt_transformer.model_configs import (
    MODEL_CONFIGS,
    get_config,
    create_full_config,
    estimate_parameters
)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    统计模型参数量
    
    Returns:
        {
            'total': 总参数量,
            'trainable': 可训练参数量,
            'non_trainable': 不可训练参数量
        }
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable
    }


def estimate_memory_usage(model: nn.Module, batch_size: int, seq_len: int, dim: int) -> Dict[str, float]:
    """
    估算模型内存占用（MB）
    
    Returns:
        {
            'model_params': 模型参数内存(MB),
            'activations': 激活值内存(MB),
            'gradients': 梯度内存(MB),
            'total': 总内存(MB)
        }
    """
    # 模型参数内存（float32，4字节）
    param_memory = sum(p.numel() * 4 for p in model.parameters()) / (1024 ** 2)
    
    # 激活值内存（前向传播）
    # 假设主要激活值在循环层和记忆系统
    activation_memory = (batch_size * seq_len * dim * 4) / (1024 ** 2)  # 简化估算
    
    # 梯度内存（与参数相同）
    gradient_memory = param_memory
    
    # 总内存（训练时）
    total_memory = param_memory + activation_memory * 2 + gradient_memory  # 激活值可能需要存储两次
    
    return {
        'model_params': param_memory,
        'activations': activation_memory,
        'gradients': gradient_memory,
        'total': total_memory
    }


def test_forward_speed(
    model: nn.Module,
    token_ids: torch.Tensor,
    num_warmup: int = 5,
    num_runs: int = 20
) -> Dict[str, float]:
    """
    测试前向传播速度
    
    Returns:
        {
            'avg_time_ms': 平均时间(毫秒),
            'min_time_ms': 最小时间(毫秒),
            'max_time_ms': 最大时间(毫秒),
            'throughput_samples_per_sec': 吞吐量(样本/秒)
        }
    """
    model.eval()
    device = next(model.parameters()).device
    token_ids = token_ids.to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(token_ids)
    
    # 同步GPU（如果有）
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 测试
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(token_ids)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # 转换为毫秒
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    batch_size = token_ids.shape[0]
    throughput = batch_size / (avg_time / 1000)
    
    return {
        'avg_time_ms': avg_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'throughput_samples_per_sec': throughput
    }


def test_training_speed(
    model: nn.Module,
    token_ids: torch.Tensor,
    num_warmup: int = 3,
    num_runs: int = 10
) -> Dict[str, float]:
    """
    测试训练速度（前向+反向传播）
    
    Returns:
        {
            'avg_time_ms': 平均时间(毫秒),
            'throughput_samples_per_sec': 吞吐量(样本/秒)
        }
    """
    model.train()
    device = next(model.parameters()).device
    token_ids = token_ids.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 预热
    for _ in range(num_warmup):
        optimizer.zero_grad()
        output, constraint_loss = model(token_ids, return_constraint_loss=True)
        loss = output.mean() + constraint_loss
        loss.backward()
        optimizer.step()
    
    # 同步GPU（如果有）
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 测试
    times = []
    for _ in range(num_runs):
        optimizer.zero_grad()
        start = time.time()
        output, constraint_loss = model(token_ids, return_constraint_loss=True)
        loss = output.mean() + constraint_loss
        loss.backward()
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)
    
    avg_time = sum(times) / len(times)
    batch_size = token_ids.shape[0]
    throughput = batch_size / (avg_time / 1000)
    
    return {
        'avg_time_ms': avg_time,
        'throughput_samples_per_sec': throughput
    }


def test_nested_learning_layers(model: nn.Module) -> Dict[str, Any]:
    """
    测试嵌套学习的层数
    
    Returns:
        {
            'omega_net_layers': ω网络的层数,
            'pi_net_layers': π网络的层数,
            'has_constraint': 是否有约束层
        }
    """
    nested_learning = model.nested_learning
    
    if nested_learning is None:
        return {
            'omega_net_layers': 0,
            'pi_net_layers': 0,
            'has_constraint': False,
            'enabled': False
        }
    
    # 统计ω网络的层数（嵌套层级数）
    omega_layers = len(nested_learning.omega_nets)
    
    # 统计π网络的层数（嵌套层级数）
    pi_layers = len(nested_learning.pi_nets)
    
    return {
        'omega_net_layers': omega_layers,
        'pi_net_layers': pi_layers,
        'has_constraint': nested_learning.incidence_constraint is not None,
        'enabled': True
    }


def test_model_architecture(model: nn.Module) -> Dict[str, Any]:
    """
    测试模型架构信息
    
    Returns:
        架构信息字典
    """
    info = {
        'vocab_size': model.vocab_size,
        'dim': model.dim,
        'hidden_dim': model.hidden_dim,
        'num_recurrent_layers': model.num_recurrent_layers,
        'num_memories': model.memory_system.num_memories,
        'use_nested_learning': model.use_nested_learning,
        'bidirectional': model.bidirectional
    }
    
    return info


def print_performance_report(
    model: nn.Module,
    token_ids: torch.Tensor,
    config: Dict[str, Any]
):
    """
    打印完整的性能报告
    """
    print("=" * 80)
    print("扭量化HOPE架构性能测试报告")
    print("=" * 80)
    
    # 1. 运行环境信息
    print("\n【1. 运行环境信息】")
    device = next(model.parameters()).device
    print(f"  设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"  GPU显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"  已用显存: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        print(f"  缓存显存: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
    
    # 2. 模型架构信息
    print("\n【2. 模型架构信息】")
    arch_info = test_model_architecture(model)
    for key, value in arch_info.items():
        print(f"  {key}: {value}")
    
    # 3. 嵌套学习层数
    print("\n【3. 嵌套学习层数】")
    nested_info = test_nested_learning_layers(model)
    if nested_info['enabled']:
        print(f"  ω网络层数: {nested_info['omega_net_layers']} 层")
        print(f"  π网络层数: {nested_info['pi_net_layers']} 层")
        print(f"  关联约束: {'启用' if nested_info['has_constraint'] else '禁用'}")
        print(f"  嵌套层级: 2层（ω和π两个分量独立优化）")
    else:
        print("  嵌套学习: 未启用")
    
    # 4. 参数量统计
    print("\n【4. 参数量统计】")
    param_info = count_parameters(model)
    print(f"  总参数量: {param_info['total']:,} ({param_info['total']/1e6:.2f}M)")
    print(f"  可训练参数: {param_info['trainable']:,} ({param_info['trainable']/1e6:.2f}M)")
    print(f"  不可训练参数: {param_info['non_trainable']:,}")
    
    # 5. 内存占用估算
    print("\n【5. 内存占用估算】")
    batch_size, seq_len = token_ids.shape
    mem_info = estimate_memory_usage(model, batch_size, seq_len, model.dim)
    print(f"  模型参数内存: {mem_info['model_params']:.2f} MB")
    print(f"  激活值内存: {mem_info['activations']:.2f} MB")
    print(f"  梯度内存: {mem_info['gradients']:.2f} MB")
    print(f"  总内存(训练): {mem_info['total']:.2f} MB")
    
    # 如果使用GPU，显示GPU显存使用情况
    if device.type == 'cuda':
        gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        print(f"  GPU显存已分配: {gpu_mem_allocated:.2f} MB")
        print(f"  GPU显存已缓存: {gpu_mem_reserved:.2f} MB")
    
    # 6. 前向传播速度
    print("\n【6. 前向传播性能】")
    forward_info = test_forward_speed(model, token_ids)
    print(f"  平均时间: {forward_info['avg_time_ms']:.2f} ms")
    print(f"  最小时间: {forward_info['min_time_ms']:.2f} ms")
    print(f"  最大时间: {forward_info['max_time_ms']:.2f} ms")
    print(f"  吞吐量: {forward_info['throughput_samples_per_sec']:.2f} 样本/秒")
    
    # 7. 训练速度
    print("\n【7. 训练性能】")
    train_info = test_training_speed(model, token_ids)
    print(f"  平均时间(前向+反向): {train_info['avg_time_ms']:.2f} ms")
    print(f"  吞吐量: {train_info['throughput_samples_per_sec']:.2f} 样本/秒")
    
    # 8. 本地训练可行性
    print("\n【8. 本地训练可行性评估】")
    total_mem_mb = mem_info['total']
    if device.type == 'cuda':
        # GPU训练评估
        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_mem_mb < 1000:
            feasibility = "✅ 完全可行（<1GB）"
        elif total_mem_mb < 2000:
            feasibility = "✅ 可行（<2GB）"
        elif total_mem_mb < 3500:
            feasibility = "✅ 可行（<3.5GB，适合GTX 1650的4GB显存）"
        else:
            feasibility = "⚠️  显存占用较高，建议减小batch size或模型规模"
        
        print(f"  GPU显存需求: {feasibility}")
        print(f"  当前配置占用: {total_mem_mb:.1f} MB ({total_mem_mb/1024:.2f} GB)")
        print(f"  GPU总显存: {gpu_total_gb:.2f} GB")
        print(f"  可用显存余量: {gpu_total_gb - total_mem_mb/1024:.2f} GB")
        
        if total_mem_mb / 1024 > gpu_total_gb * 0.9:
            print("  ⚠️  警告: 显存占用超过90%，建议减小batch size")
    else:
        # CPU训练评估
        if total_mem_mb < 1000:
            feasibility = "✅ 完全可行（<1GB）"
        elif total_mem_mb < 4000:
            feasibility = "✅ 可行（<4GB，需要中等配置）"
        elif total_mem_mb < 8000:
            feasibility = "⚠️  需要较高配置（<8GB，建议16GB内存）"
        else:
            feasibility = "❌ 需要高配置（>8GB，建议32GB+内存）"
        
        print(f"  内存需求: {feasibility}")
        print(f"  推荐配置: CPU训练需要 {total_mem_mb*2:.0f}MB+ 内存")
    
    # 9. 性能指标总结
    print("\n【9. 性能指标总结】")
    print(f"  模型规模: {param_info['total']/1e6:.2f}M 参数")
    print(f"  训练速度: {train_info['throughput_samples_per_sec']:.1f} 样本/秒")
    print(f"  内存占用: {total_mem_mb:.1f} MB")
    print(f"  复杂度: O(n) - 线性复杂度（Recurrent结构）")
    
    # 10. GPU优化建议（如果使用GPU）
    if device.type == 'cuda':
        print("\n【10. GPU训练优化建议】")
        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        current_batch_size = token_ids.shape[0]
        
        if "1650" in torch.cuda.get_device_name(0) or gpu_total_gb <= 4.5:
            print("  检测到GTX 1650 (4GB显存) 或类似配置")
            print("  优化建议:")
            
            if total_mem_mb / 1024 > 3.0:
                recommended_batch = max(1, current_batch_size // 2)
                print(f"  • 当前显存占用较高，建议减小batch size: {current_batch_size} → {recommended_batch}")
            
            if param_info['total']/1e6 > 10:
                print(f"  • 模型参数量较大 ({param_info['total']/1e6:.1f}M)，可考虑:")
                print("    - 减少 num_recurrent_layers")
                print("    - 减少 num_memories")
                print("    - 减少 dim 维度")
            
            print("  • 使用混合精度训练 (torch.cuda.amp) 可节省约50%显存")
            print("  • 使用梯度累积可以等效增大batch size而不增加显存")
            print("  • 定期调用 torch.cuda.empty_cache() 清理未使用的显存缓存")
        else:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  当前显存使用率: {(total_mem_mb / 1024) / gpu_total_gb * 100:.1f}%")
    
    print("\n" + "=" * 80)


def main():
    """主函数"""
    # 检测GPU设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
        print(f"检测到GPU: {gpu_name}")
        print(f"GPU显存: {gpu_memory:.2f} GB")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"PyTorch版本: {torch.__version__}")
    else:
        device = torch.device('cpu')
        print("未检测到GPU，使用CPU运行")
        gpu_name = None
        gpu_memory = None
    
    print("-" * 80)
    
    # 配置
    config = {
        'vocab_size': 1000,
        'dim': 128,
        'hidden_dim': 128,
        'num_recurrent_layers': 2,
        'num_memories': 3,
        'num_memory_cycles': 2,
        'use_nested_learning': True,
        'use_phase_compression': True,
        'bidirectional': False,
        'dropout': 0.1,
        'batch_size': 8 if torch.cuda.is_available() else 4,  # GPU可以使用更大的batch size
        'seq_len': 32,
        'device': device
    }
    
    # 创建模型
    print("正在创建模型...")
    model = TwistorHopeArchitecture(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        hidden_dim=config['hidden_dim'],
        num_recurrent_layers=config['num_recurrent_layers'],
        num_memories=config['num_memories'],
        num_memory_cycles=config['num_memory_cycles'],
        use_nested_learning=config['use_nested_learning'],
        use_phase_compression=config['use_phase_compression'],
        bidirectional=config['bidirectional'],
        dropout=config['dropout']
    )
    
    # 将模型移动到设备
    model = model.to(device)
    print(f"模型已移动到: {device}")
    
    # 创建测试数据并移动到设备
    token_ids = torch.randint(0, config['vocab_size'], 
                             (config['batch_size'], config['seq_len']),
                             device=device)
    
    # 打印性能报告
    print_performance_report(model, token_ids, config)
    
    # 测试不同配置
    print("\n" + "=" * 80)
    print("测试不同配置的性能对比")
    print("=" * 80)
    
    # 使用预设配置
    config_names = ['small', 'medium', 'recommended', 'large']
    results = []
    
    for config_name in config_names:
        print(f"\n【{config_name.upper()} 配置】")
        try:
            # 获取配置
            base_config = get_config(config_name)
            full_config = create_full_config(
                config_name=config_name,
                vocab_size=1000
            )
            
            # 创建模型
            test_model = TwistorHopeArchitecture(
                vocab_size=full_config['vocab_size'],
                dim=full_config['dim'],
                hidden_dim=full_config['hidden_dim'],
                num_recurrent_layers=full_config['num_recurrent_layers'],
                num_memories=full_config['num_memories'],
                num_memory_cycles=full_config['num_memory_cycles'],
                use_nested_learning=full_config['use_nested_learning'],
                use_phase_compression=full_config['use_phase_compression'],
                bidirectional=full_config['bidirectional'],
                dropout=full_config['dropout'],
                num_mobius_cycles=full_config['num_mobius_cycles'],
                use_adaptive_evolution_rate=full_config['use_adaptive_evolution_rate'],
                use_multiscale_evolution=full_config['use_multiscale_evolution'],
                num_nested_levels=full_config['num_nested_levels'],
                use_level_constraints=full_config['use_level_constraints']
            )
            test_model = test_model.to(device)
            
            # 测试性能
            param_info = count_parameters(test_model)
            mem_info = estimate_memory_usage(test_model, 4, 32, full_config['dim'])
            
            # 测试前向传播速度
            test_token_ids = torch.randint(
                0, full_config['vocab_size'],
                (4, 32),
                device=device
            )
            forward_info = test_forward_speed(test_model, test_token_ids, num_runs=10)
            
            # 保存结果
            result = {
                'config_name': config_name,
                'dim': full_config['dim'],
                'num_recurrent_layers': full_config['num_recurrent_layers'],
                'num_memories': full_config['num_memories'],
                'bidirectional': full_config['bidirectional'],
                'num_nested_levels': full_config['num_nested_levels'],
                'params': param_info['total'],
                'memory_mb': mem_info['total'],
                'forward_time_ms': forward_info['avg_time_ms']
            }
            results.append(result)
            
            # 打印结果
            print(f"  维度: {full_config['dim']}")
            print(f"  循环层数: {full_config['num_recurrent_layers']}")
            print(f"  记忆数量: {full_config['num_memories']}")
            print(f"  双向循环: {full_config['bidirectional']}")
            print(f"  嵌套层级: {full_config['num_nested_levels']}")
            print(f"  参数量: {param_info['total']/1e6:.2f}M")
            print(f"  内存: {mem_info['total']:.1f} MB")
            print(f"  前向传播: {forward_info['avg_time_ms']:.2f} ms")
            
            # 清理显存
            if torch.cuda.is_available():
                del test_model, test_token_ids
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            continue
    
    # 打印对比表格
    print("\n" + "=" * 80)
    print("配置性能对比总结")
    print("=" * 80)
    print(f"{'配置':<12} {'参数量(M)':<12} {'内存(MB)':<12} {'前向(ms)':<12} {'双向':<8}")
    print("-" * 80)
    for result in results:
        print(f"{result['config_name']:<12} "
              f"{result['params']/1e6:<12.2f} "
              f"{result['memory_mb']:<12.1f} "
              f"{result['forward_time_ms']:<12.2f} "
              f"{'是' if result['bidirectional'] else '否':<8}")
    
    # 测试推荐配置的详细性能
    print("\n" + "=" * 80)
    print("推荐配置详细性能测试")
    print("=" * 80)
    test_recommended_config(device)


def test_recommended_config(device: torch.device):
    """
    测试推荐配置的详细性能
    
    Args:
        device: 设备
    """
    print("\n创建推荐配置模型...")
    config_name = 'recommended'
    full_config = create_full_config(
        config_name=config_name,
        vocab_size=1000
    )
    
    model = TwistorHopeArchitecture(
        vocab_size=full_config['vocab_size'],
        dim=full_config['dim'],
        hidden_dim=full_config['hidden_dim'],
        num_recurrent_layers=full_config['num_recurrent_layers'],
        num_memories=full_config['num_memories'],
        num_memory_cycles=full_config['num_memory_cycles'],
        use_nested_learning=full_config['use_nested_learning'],
        use_phase_compression=full_config['use_phase_compression'],
        bidirectional=full_config['bidirectional'],
        dropout=full_config['dropout'],
        num_mobius_cycles=full_config['num_mobius_cycles'],
        use_adaptive_evolution_rate=full_config['use_adaptive_evolution_rate'],
        use_multiscale_evolution=full_config['use_multiscale_evolution'],
        num_nested_levels=full_config['num_nested_levels'],
        use_level_constraints=full_config['use_level_constraints']
    )
    model = model.to(device)
    
    # 创建测试数据
    batch_size = 8
    seq_len = 32
    token_ids = torch.randint(
        0, full_config['vocab_size'],
        (batch_size, seq_len),
        device=device
    )
    
    # 打印详细性能报告
    print_performance_report(model, token_ids, full_config)
    
    # 清理
    if torch.cuda.is_available():
        del model, token_ids
        torch.cuda.empty_cache()


def compare_configs():
    """
    对比不同配置的性能
    测试small, recommended, large配置
    """
    print("=" * 80)
    print("配置性能对比")
    print("=" * 80)
    
    # 检测设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    config_names = ['small', 'recommended', 'large']
    
    for config_name in config_names:
        print(f"\n【{config_name.upper()} 配置】")
        try:
            full_config = create_full_config(
                config_name=config_name,
                vocab_size=1000
            )
            
            model = TwistorHopeArchitecture(
                vocab_size=full_config['vocab_size'],
                dim=full_config['dim'],
                hidden_dim=full_config['hidden_dim'],
                num_recurrent_layers=full_config['num_recurrent_layers'],
                num_memories=full_config['num_memories'],
                use_nested_learning=full_config['use_nested_learning'],
                bidirectional=full_config['bidirectional'],
                num_nested_levels=full_config['num_nested_levels']
            )
            model = model.to(device)
            
            param_info = count_parameters(model)
            print(f"  参数量: {param_info['total']/1e6:.2f}M")
            
            if torch.cuda.is_available():
                del model
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")


if __name__ == '__main__':
    main()

