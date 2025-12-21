"""
扭量化HOPE架构 - 优化训练脚本
集成所有训练优化：混合精度、梯度累积、学习率调度、配置选择
"""

import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
import time
import sys
import os
import argparse
from typing import Dict, Any, Optional
from tqdm import tqdm

# 添加父目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mt_transformer import TwistorHopeArchitecture
from mt_transformer.model_configs import (
    get_config,
    create_full_config,
    estimate_parameters,
    validate_config
)
from examples.train_gpu import (
    setup_device,
    count_parameters,
    create_dummy_dataset,
    train_epoch,
    validate,
    print_gpu_memory,
    WarmupCosineScheduler
)


def create_optimized_model(
    config_name: str = 'recommended',
    vocab_size: int = 1000,
    device: torch.device = None
) -> nn.Module:
    """
    创建优化配置的模型
    
    Args:
        config_name: 配置名称 ('small', 'medium', 'recommended', 'large')
        vocab_size: 词汇表大小
        device: 设备
    
    Returns:
        模型实例
    """
    # 获取完整配置
    full_config = create_full_config(
        config_name=config_name,
        vocab_size=vocab_size
    )
    
    # 验证配置
    validate_config(full_config)
    
    # 创建模型
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
        nested_level_lrs=full_config['nested_level_lrs'],
        use_level_constraints=full_config['use_level_constraints']
    )
    
    if device is not None:
        model = model.to(device)
    
    # 创建嵌套优化器（如果使用嵌套学习）
    if full_config.get('use_nested_learning', True) and model.nested_learning is not None:
        model.nested_learning.create_nested_optimizers()
    
    return model, full_config


def train_optimized(
    model: nn.Module,
    train_loader,
    val_loader,
    config: Dict[str, Any],
    device: torch.device,
    use_gpu: bool
):
    """
    优化的训练循环
    
    集成功能：
    - 混合精度训练
    - 梯度累积
    - 学习率调度（Warmup + Cosine）
    - 层级学习率（如果使用嵌套学习）
    """
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # 混合精度训练
    scaler = None
    if config.get('use_amp', use_gpu):
        try:
            scaler = GradScaler('cuda')
        except TypeError:
            scaler = GradScaler()  # 兼容旧版本
    
    # 学习率调度器
    lr_scheduler = None
    if config.get('use_lr_scheduling', True):
        num_batches_per_epoch = config['num_train_samples'] // config['batch_size']
        total_steps = num_batches_per_epoch * config['num_epochs']
        warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))
        
        lr_scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            max_lr=config['learning_rate'],
            min_lr=config.get('min_lr', 1e-6)
        )
    
    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("\n开始训练...")
    print("=" * 80)
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        
        # 训练
        train_loss, train_constraint_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            use_amp=config.get('use_amp', use_gpu),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            clip_grad_norm=config.get('clip_grad_norm', None),
            lr_scheduler=lr_scheduler,
            current_epoch=epoch,
            skip_nan=config.get('skip_nan', True)
        )
        
        # 更新嵌套学习的训练进度
        if config.get('use_nested_learning', True) and model.nested_learning is not None:
            num_batches_per_epoch = config['num_train_samples'] // config['batch_size']
            current_step = epoch * num_batches_per_epoch + len(list(train_loader))
            total_steps = num_batches_per_epoch * config['num_epochs']
            model.nested_learning.set_training_progress(current_step, total_steps)
        
        # 验证
        val_loss = validate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start_time
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 打印训练信息
        current_lr = lr_scheduler.get_last_lr() if lr_scheduler is not None else config['learning_rate']
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] ({epoch_time:.2f}s)")
        print(f"  训练损失: {train_loss:.6f} (约束损失: {train_constraint_loss:.6f})")
        print(f"  验证损失: {val_loss:.6f}")
        print(f"  学习率: {current_lr:.2e}")
        print_gpu_memory(device)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f"best_model_{config.get('config_name', 'optimized')}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, save_path)
            print(f"  ✓ 保存最佳模型 (验证损失: {val_loss:.6f})")
        
        print()
        
        # 清理显存缓存
        if use_gpu:
            torch.cuda.empty_cache()
    
    print("=" * 80)
    print("训练完成！")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    return train_losses, val_losses


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='扭量化HOPE架构 - 优化训练脚本')
    parser.add_argument(
        '--config',
        type=str,
        default='recommended',
        choices=['small', 'medium', 'recommended', 'large'],
        help='模型配置 (default: recommended)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=1000,
        help='词汇表大小 (default: 1000)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (默认根据配置自动选择)'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help='训练轮数 (default: 10)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='学习率 (默认根据配置自动选择)'
    )
    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='禁用混合精度训练'
    )
    parser.add_argument(
        '--no-lr-scheduling',
        action='store_true',
        help='禁用学习率调度'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("扭量化HOPE架构 - 优化训练脚本")
    print("=" * 80)
    print()
    
    # 设置设备
    device, use_gpu = setup_device()
    print()
    
    # 创建模型
    print(f"创建模型 (配置: {args.config})...")
    model, full_config = create_optimized_model(
        config_name=args.config,
        vocab_size=args.vocab_size,
        device=device
    )
    
    # 统计参数量
    num_params = count_parameters(model)
    estimated_params = estimate_parameters(full_config)
    print(f"✓ 模型创建完成")
    print(f"  实际参数量: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"  估算参数量: {estimated_params:,} ({estimated_params/1e6:.2f}M)")
    print_gpu_memory(device)
    print()
    
    # 训练配置
    config = full_config.copy()
    config.update({
        'config_name': args.config,
        'batch_size': args.batch_size if args.batch_size else (16 if use_gpu else 4),
        'seq_len': 32,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate if args.learning_rate else 1e-4,
        'min_lr': 1e-6,
        'gradient_accumulation_steps': 4 if use_gpu else 2,
        'clip_grad_norm': 1.0,
        'use_amp': use_gpu and not args.no_amp,
        'use_lr_scheduling': not args.no_lr_scheduling,
        'warmup_ratio': 0.1,
        'skip_nan': True,
        'weight_decay': 1e-5,
        'num_train_samples': 1000,
        'num_val_samples': 200,
    })
    
    print("训练配置:")
    for key, value in config.items():
        if key not in ['vocab_size', 'dim', 'hidden_dim', 'num_recurrent_layers',
                       'num_memories', 'num_memory_cycles', 'use_nested_learning',
                       'use_phase_compression', 'bidirectional', 'dropout',
                       'num_mobius_cycles', 'use_adaptive_evolution_rate',
                       'use_multiscale_evolution', 'num_nested_levels',
                       'nested_level_lrs', 'use_level_constraints']:
            print(f"  {key}: {value}")
    print()
    
    # 创建数据集
    print("创建数据集...")
    train_data, train_targets = create_dummy_dataset(
        config['vocab_size'],
        config['num_train_samples'],
        config['seq_len'],
        device
    )
    val_data, val_targets = create_dummy_dataset(
        config['vocab_size'],
        config['num_val_samples'],
        config['seq_len'],
        device
    )
    
    # 创建数据加载器
    def create_loader(data, targets, batch_size, shuffle=True):
        num_batches = len(data) // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            yield data[start:end], targets[start:end]
    
    print(f"✓ 训练集: {len(train_data)} 样本")
    print(f"✓ 验证集: {len(val_data)} 样本")
    print()
    
    # 训练
    train_loader = create_loader(train_data, train_targets, config['batch_size'], shuffle=True)
    val_loader = create_loader(val_data, val_targets, config['batch_size'], shuffle=False)
    
    train_losses, val_losses = train_optimized(
        model,
        train_loader,
        val_loader,
        config,
        device,
        use_gpu
    )
    
    # 最终显存使用统计
    if use_gpu:
        print("\n最终显存使用:")
        print_gpu_memory(device)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ GPU显存不足！")
            print("建议:")
            print("  1. 使用更小的配置 (--config small)")
            print("  2. 减小 batch size (--batch-size)")
            print("  3. 增加梯度累积步数")
            print("  4. 确保启用混合精度训练 (不要使用 --no-amp)")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise e

