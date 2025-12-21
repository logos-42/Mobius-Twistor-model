"""
扭量化HOPE架构 - GPU训练脚本
针对GTX 1650 (4GB显存) 优化
支持混合精度训练和梯度累积
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
import math
from typing import Dict, Any, Optional
from tqdm import tqdm

# 添加父目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mt_transformer import TwistorHopeArchitecture


class WarmupCosineScheduler:
    """
    Warmup + Cosine退火学习率调度器
    前warmup_steps线性增加，之后cosine衰减
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        max_lr: float,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self) -> float:
        """更新学习率"""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Warmup阶段：线性增加
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine退火阶段
            progress = (self.current_step - self.warmup_steps) / \
                      max(self.total_steps - self.warmup_steps, 1)
            lr = self.min_lr + (self.max_lr - self.min_lr) * \
                 (1 + math.cos(math.pi * progress)) / 2
        
        # 更新所有参数组的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


def setup_device():
    """设置设备并检查GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"✓ 检测到GPU: {gpu_name}")
        print(f"✓ GPU显存: {gpu_memory:.2f} GB")
        print(f"✓ CUDA版本: {torch.version.cuda}")
        print(f"✓ PyTorch版本: {torch.__version__}")
        return device, True
    else:
        print("⚠️  未检测到GPU，使用CPU运行（训练会很慢）")
        return torch.device('cpu'), False


def create_model(config: Dict[str, Any], device: torch.device):
    """创建模型并移动到设备"""
    model = TwistorHopeArchitecture(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        hidden_dim=config.get('hidden_dim', config['dim']),
        num_recurrent_layers=config['num_recurrent_layers'],
        num_memories=config['num_memories'],
        num_memory_cycles=config.get('num_memory_cycles', 2),
        use_nested_learning=config.get('use_nested_learning', True),
        use_phase_compression=config.get('use_phase_compression', True),
        bidirectional=config.get('bidirectional', False),
        dropout=config.get('dropout', 0.1),
        num_nested_levels=config.get('num_nested_levels', 5)
    )
    model = model.to(device)
    
    # 创建嵌套优化器（如果使用嵌套学习）
    if config.get('use_nested_learning', True) and model.nested_learning is not None:
        model.nested_learning.create_nested_optimizers()
    
    return model


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters())


def create_dummy_dataset(vocab_size: int, num_samples: int, seq_len: int, device: torch.device):
    """创建虚拟数据集（用于演示）"""
    # 在实际使用中，应该替换为真实的数据加载器
    data = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)
    return data, targets


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: Optional[float] = None,
    lr_scheduler: Optional[WarmupCosineScheduler] = None,
    current_epoch: int = 0,
    skip_nan: bool = True
):
    """
    训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 设备
        scaler: 梯度缩放器（用于混合精度）
        use_amp: 是否使用混合精度
        gradient_accumulation_steps: 梯度累积步数
        clip_grad_norm: 梯度裁剪阈值
        lr_scheduler: 学习率调度器
        current_epoch: 当前epoch
        skip_nan: 是否跳过NaN损失
    """
    model.train()
    total_loss = 0.0
    total_constraint_loss = 0.0
    num_batches = 0
    skipped_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (token_ids, targets) in enumerate(train_loader):
        token_ids = token_ids.to(device)
        targets = targets.to(device)
        
        # 混合精度训练
        if use_amp and scaler is not None:
            try:
                autocast_context = autocast('cuda')
            except TypeError:
                autocast_context = autocast()  # 兼容旧版本
            
            with autocast_context:
                output, constraint_loss = model(token_ids, return_constraint_loss=True)
                # 获取目标embedding并投影到输出维度
                target_emb = model.embedding(targets)  # (batch, seq, dim*2)
                # 投影到输出维度：取前半部分（omega分量）或使用平均
                target_proj = target_emb[..., :output.shape[-1]]  # 取前dim维
                # 简单的语言模型损失（实际应用中应使用交叉熵等）
                loss = nn.functional.mse_loss(output, target_proj.detach()) + constraint_loss
                loss = loss / gradient_accumulation_steps
            
            # NaN检测
            if skip_nan and (torch.isnan(loss) or torch.isinf(loss)):
                skipped_batches += 1
                optimizer.zero_grad()
                continue
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪（在unscale之后）
                if clip_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    # 再次检查NaN（unscale后可能产生NaN）
                    if skip_nan:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), clip_grad_norm
                        )
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            skipped_batches += 1
                            scaler.update()
                            optimizer.zero_grad()
                            continue
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # 更新学习率
                if lr_scheduler is not None:
                    lr_scheduler.step()
        else:
            # 普通精度训练
            output, constraint_loss = model(token_ids, return_constraint_loss=True)
            # 获取目标embedding并投影到输出维度
            target_emb = model.embedding(targets)  # (batch, seq, dim*2)
            # 投影到输出维度：取前半部分（omega分量）
            target_proj = target_emb[..., :output.shape[-1]]  # 取前dim维
            loss = nn.functional.mse_loss(output, target_proj.detach()) + constraint_loss
            loss = loss / gradient_accumulation_steps
            
            # NaN检测
            if skip_nan and (torch.isnan(loss) or torch.isinf(loss)):
                skipped_batches += 1
                optimizer.zero_grad()
                continue
            
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪
                if clip_grad_norm is not None:
                    if skip_nan:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), clip_grad_norm
                        )
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            skipped_batches += 1
                            optimizer.zero_grad()
                            continue
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # 更新学习率
                if lr_scheduler is not None:
                    lr_scheduler.step()
        
        total_loss += loss.item() * gradient_accumulation_steps
        total_constraint_loss += constraint_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_constraint_loss = total_constraint_loss / num_batches if num_batches > 0 else 0.0
    
    if skipped_batches > 0:
        print(f"  警告: 跳过了 {skipped_batches} 个包含NaN/Inf的batch")
    
    return avg_loss, avg_constraint_loss


def validate(model: nn.Module, val_loader, device: torch.device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for token_ids, targets in val_loader:
            token_ids = token_ids.to(device)
            targets = targets.to(device)
            
            output, constraint_loss = model(token_ids, return_constraint_loss=True)
            target_emb = model.embedding(targets)  # (batch, seq, dim*2)
            target_proj = target_emb[..., :output.shape[-1]]  # 取前dim维
            loss = nn.functional.mse_loss(output, target_proj) + constraint_loss
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def print_gpu_memory(device: torch.device):
    """打印GPU显存使用情况"""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        print(f"  显存 - 已分配: {allocated:.1f} MB, 已缓存: {reserved:.1f} MB")


def main():
    """主训练函数"""
    print("=" * 80)
    print("扭量化HOPE架构 - GPU训练脚本")
    print("=" * 80)
    print()
    
    # 设置设备
    device, use_gpu = setup_device()
    print()
    
    # 训练配置（针对GTX 1650优化）
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
        'num_nested_levels': 5,
        
        # 训练超参数
        'batch_size': 8 if use_gpu else 2,  # GPU可以使用更大的batch size
        'seq_len': 32,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'min_lr': 1e-6,  # 最小学习率（用于cosine退火）
        'gradient_accumulation_steps': 2,  # 梯度累积，等效batch size = 16
        'clip_grad_norm': 1.0,
        
        # 混合精度训练（节省显存）
        'use_amp': use_gpu,  # GPU上自动启用混合精度
        
        # 学习率调度
        'use_lr_scheduling': True,  # 启用学习率调度
        'warmup_ratio': 0.1,  # Warmup占总步数的比例
        
        # NaN处理
        'skip_nan': True,  # 跳过包含NaN/Inf的batch
        
        # 数据集配置
        'num_train_samples': 1000,
        'num_val_samples': 200,
    }
    
    print("训练配置:")
    for key, value in config.items():
        if 'num' in key.lower() or 'batch' in key.lower() or 'epoch' in key.lower():
            print(f"  {key}: {value}")
    print()
    
    # 创建模型
    print("创建模型...")
    model = create_model(config, device)
    num_params = count_parameters(model)
    print(f"✓ 模型创建完成，参数量: {num_params:,} ({num_params/1e6:.2f}M)")
    print_gpu_memory(device)
    print()
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    
    # 混合精度训练
    if config['use_amp']:
        try:
            scaler = GradScaler('cuda')
        except TypeError:
            scaler = GradScaler()  # 兼容旧版本
    else:
        scaler = None
    if config['use_amp']:
        print("✓ 混合精度训练已启用（可节省约50%显存）")
    
    # 学习率调度器
    lr_scheduler = None
    if config.get('use_lr_scheduling', True):
        # 计算总步数
        num_batches_per_epoch = config['num_train_samples'] // config['batch_size']
        total_steps = num_batches_per_epoch * config['num_epochs']
        warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))  # 默认10%用于warmup
        
        lr_scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            max_lr=config['learning_rate'],
            min_lr=config.get('min_lr', 1e-6)
        )
        print(f"✓ 学习率调度器已启用 (Warmup: {warmup_steps}步, 总步数: {total_steps})")
    
    # 创建数据集
    print("\n创建数据集...")
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
    
    # 创建数据加载器（简化版本）
    def create_loader(data, targets, batch_size, shuffle=True):
        num_batches = len(data) // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            yield data[start:end], targets[start:end]
    
    print(f"✓ 训练集: {len(train_data)} 样本")
    print(f"✓ 验证集: {len(val_data)} 样本")
    print()
    
    # 训练循环
    print("开始训练...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        
        # 创建数据加载器
        train_loader = create_loader(train_data, train_targets, config['batch_size'], shuffle=True)
        val_loader = create_loader(val_data, val_targets, config['batch_size'], shuffle=False)
        
        # 训练
        train_loss, train_constraint_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            use_amp=config['use_amp'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            clip_grad_norm=config['clip_grad_norm'],
            lr_scheduler=lr_scheduler,
            current_epoch=epoch,
            skip_nan=config.get('skip_nan', True)
        )
        
        # 更新嵌套学习的训练进度（如果使用嵌套学习）
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, 'best_model.pth')
            print(f"  ✓ 保存最佳模型 (验证损失: {val_loss:.6f})")
        
        print()
        
        # 清理显存缓存
        if use_gpu:
            torch.cuda.empty_cache()
    
    print("=" * 80)
    print("训练完成！")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"模型已保存到: best_model.pth")
    
    # 显存使用统计
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
            print("  1. 减小 batch_size（当前配置中）")
            print("  2. 减小 seq_len")
            print("  3. 减小 dim 或 num_recurrent_layers")
            print("  4. 增加 gradient_accumulation_steps")
            print("  5. 确保 use_amp=True（混合精度训练）")
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise e
