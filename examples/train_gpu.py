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
from torch.utils.data import Dataset, DataLoader

# HuggingFace相关导入
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  警告: 未安装HuggingFace库，将使用虚拟数据集")
    print("   安装命令: pip install datasets transformers")

# 添加父目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mt_transformer import TwistorHopeArchitecture
from mt_transformer.model_configs import get_config, create_full_config


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


class HFDataset(Dataset):
    """HuggingFace数据集包装器"""
    
    def __init__(self, hf_dataset, tokenizer, seq_len: int, text_column: str = 'text'):
        """
        Args:
            hf_dataset: HuggingFace数据集
            tokenizer: 分词器
            seq_len: 序列长度
            text_column: 文本列名
        """
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_column = text_column
        
        # 预处理数据集：过滤空文本并分词
        print("  正在预处理数据集...")
        self.processed_data = []
        
        for item in tqdm(self.dataset, desc="  处理数据"):
            text = item.get(self.text_column, '')
            if text and isinstance(text, str) and len(text.strip()) > 0:
                # 分词
                tokens = self.tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    max_length=seq_len + 1,
                    truncation=True,
                    padding='max_length'
                )
                
                if len(tokens) >= 2:  # 至少需要2个token才能创建输入-目标对
                    self.processed_data.append(tokens)
        
        print(f"  ✓ 处理完成，有效样本数: {len(self.processed_data)}")
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        tokens = self.processed_data[idx]
        # 输入是前seq_len个token，目标是后seq_len个token（右移1位）
        data = torch.tensor(tokens[:self.seq_len], dtype=torch.long)
        targets = torch.tensor(tokens[1:self.seq_len+1], dtype=torch.long)
        return data, targets


def load_hf_dataset(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    tokenizer_name: str = 'gpt2',
    seq_len: int = 32,
    text_column: str = 'text',
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None
):
    """
    加载HuggingFace数据集
    
    Args:
        dataset_name: 数据集名称（如 'wikitext', 'bookcorpus'）
        dataset_config: 数据集配置（如 'wikitext-2-raw-v1'）
        tokenizer_name: tokenizer名称或路径
        seq_len: 序列长度
        text_column: 文本列名
        max_train_samples: 最大训练样本数（None表示使用全部）
        max_val_samples: 最大验证样本数（None表示使用全部）
    
    Returns:
        train_dataset, val_dataset, tokenizer, vocab_size
    """
    if not HF_AVAILABLE:
        raise ImportError("需要安装HuggingFace库: pip install datasets transformers")
    
    print(f"加载HuggingFace数据集: {dataset_name}")
    if dataset_config:
        print(f"  配置: {dataset_config}")
    
    # 加载数据集
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        dataset = load_dataset(dataset_name)
    
    # 加载tokenizer
    print(f"加载tokenizer: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"  ⚠️  无法加载tokenizer {tokenizer_name}，尝试使用gpt2")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # 设置pad_token（如果不存在）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 获取训练集和验证集
    train_split = dataset.get('train', None)
    val_split = dataset.get('validation', dataset.get('test', None))
    
    if train_split is None:
        raise ValueError(f"数据集 {dataset_name} 没有训练集")
    
    # 限制样本数（如果指定）
    if max_train_samples:
        train_split = train_split.select(range(min(max_train_samples, len(train_split))))
    if max_val_samples and val_split:
        val_split = val_split.select(range(min(max_val_samples, len(val_split))))
    
    # 创建数据集
    print("创建训练集...")
    train_dataset = HFDataset(train_split, tokenizer, seq_len, text_column)
    
    if val_split:
        print("创建验证集...")
        val_dataset = HFDataset(val_split, tokenizer, seq_len, text_column)
    else:
        print("  ⚠️  没有验证集，将使用训练集的一部分作为验证集")
        # 从训练集中分割一部分作为验证集
        val_size = min(len(train_dataset) // 10, 1000)  # 10%或最多1000个样本
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    vocab_size = len(tokenizer)
    print(f"✓ 词汇表大小: {vocab_size}")
    
    return train_dataset, val_dataset, tokenizer, vocab_size


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
    import sys
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='扭量化HOPE架构 - GPU训练脚本')
    parser.add_argument('--config', type=str, default='recommended', 
                        help='模型配置名称 (default: recommended)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数 (default: 50, 建议30-100)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从checkpoint恢复训练 (模型文件路径, 如: best_model.pth, 默认自动检测)')
    parser.add_argument('--no-resume', action='store_true',
                        help='禁用自动恢复，强制从头开始训练')
    
    # 数据集相关参数
    parser.add_argument('--dataset', type=str, default=None,
                        help='HuggingFace数据集名称 (如: wikitext, bookcorpus, 默认使用虚拟数据集)')
    parser.add_argument('--dataset-config', type=str, default=None,
                        help='数据集配置 (如: wikitext-2-raw-v1)')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        help='Tokenizer名称或路径 (default: gpt2)')
    parser.add_argument('--text-column', type=str, default='text',
                        help='文本列名 (default: text)')
    parser.add_argument('--max-train-samples', type=int, default=None,
                        help='最大训练样本数 (None表示使用全部)')
    parser.add_argument('--max-val-samples', type=int, default=None,
                        help='最大验证样本数 (None表示使用全部)')
    
    args = parser.parse_args()
    
    config_name = args.config
    num_epochs = args.epochs
    resume_path = args.resume
    
    # 如果没有指定resume路径且未禁用自动恢复，则自动检测checkpoint文件
    if resume_path is None and not args.no_resume:
        default_checkpoint = 'best_model.pth'
        if os.path.exists(default_checkpoint):
            resume_path = default_checkpoint
    elif args.no_resume:
        resume_path = None
    
    print("=" * 80)
    print("扭量化HOPE架构 - GPU训练脚本")
    print(f"使用配置: {config_name.upper()}")
    print(f"训练轮数: {num_epochs}")
    if resume_path:
        print(f"✓ 从checkpoint恢复训练: {resume_path}")
    else:
        print("✓ 从头开始训练")
    print("=" * 80)
    print()
    
    # 设置设备
    device, use_gpu = setup_device()
    print()
    
    # 加载数据集（如果指定）
    use_hf_dataset = args.dataset is not None and HF_AVAILABLE
    tokenizer = None
    vocab_size = 1000  # 默认值
    default_seq_len = 32  # 默认序列长度
    
    if use_hf_dataset:
        try:
            train_dataset, val_dataset, tokenizer, vocab_size = load_hf_dataset(
                dataset_name=args.dataset,
                dataset_config=args.dataset_config,
                tokenizer_name=args.tokenizer,
                seq_len=default_seq_len,  # 使用默认值，后续会在config中统一
                text_column=args.text_column,
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples
            )
            print(f"✓ 成功加载HuggingFace数据集")
            print()
        except Exception as e:
            print(f"❌ 加载HuggingFace数据集失败: {e}")
            print("   将使用虚拟数据集")
            use_hf_dataset = False
    
    # 使用预设配置
    base_model_config = create_full_config(
        config_name=config_name,
        vocab_size=vocab_size,  # 使用从数据集获取的vocab_size
        num_memory_cycles=2,
        use_nested_learning=True,
        use_phase_compression=True,
        dropout=0.1
    )
    
    # 根据模型大小调整batch size（recommended配置较大，需要更小的batch size）
    model_dim = base_model_config.get('dim', 128)
    if use_gpu:
        if model_dim >= 256:
            batch_size = 4  # 大模型使用小batch
        else:
            batch_size = 8
    else:
        batch_size = 2
    
    # 训练配置（针对GTX 1650优化）
    config = {
        # 训练超参数
        'batch_size': batch_size,
        'seq_len': 32,
        'num_epochs': num_epochs,
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
    
    # 合并模型配置
    config.update(base_model_config)
    
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
    
    # 从checkpoint恢复训练（如果指定）
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint = None
    if resume_path and os.path.exists(resume_path):
        print(f"从checkpoint加载: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复训练状态
        start_epoch = checkpoint.get('epoch', 0) + 1  # 从下一个epoch开始
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"✓ 已恢复训练状态 (从epoch {start_epoch} 开始, 最佳验证损失: {best_val_loss:.6f})")
        
        # 如果checkpoint中有配置，确保配置一致（可选）
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            # 只更新epoch数，保持其他配置不变
            if saved_config.get('num_epochs') != config.get('num_epochs'):
                print(f"  注意: checkpoint中的epoch数 ({saved_config.get('num_epochs')}) 与当前设置 ({config.get('num_epochs')}) 不同")
        print()
    
    # 混合精度训练
    scaler = None
    if config['use_amp']:
        try:
            scaler = GradScaler('cuda')
        except TypeError:
            scaler = GradScaler()  # 兼容旧版本
        
        # 从checkpoint恢复scaler状态（如果存在）
        if checkpoint is not None and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("✓ 已恢复混合精度训练scaler状态")
        
        print("✓ 混合精度训练已启用（可节省约50%显存）")
    
    # 创建数据集和数据加载器（学习率调度器将在数据集创建后初始化，以便使用正确的样本数）
    print("\n准备数据集...")
    if use_hf_dataset:
        # 使用HuggingFace数据集
        # 注意：train_dataset和val_dataset已经在上面创建了
        # 如果config中的seq_len与创建时不同，需要重新创建（这里简化处理，使用创建时的seq_len）
        if config['seq_len'] != default_seq_len:
            print(f"  注意: 数据集使用序列长度 {default_seq_len}，配置中为 {config['seq_len']}")
            print(f"  将使用数据集创建时的序列长度 {default_seq_len}")
            config['seq_len'] = default_seq_len  # 统一使用数据集创建时的seq_len
        
        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,  # Windows上建议设为0
            pin_memory=use_gpu
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=use_gpu
        )
        
        # 更新配置中的样本数（用于学习率调度器计算）
        config['num_train_samples'] = len(train_dataset)
        config['num_val_samples'] = len(val_dataset)
        
        print(f"✓ 训练集: {len(train_dataset)} 样本")
        print(f"✓ 验证集: {len(val_dataset)} 样本")
    else:
        # 使用虚拟数据集（向后兼容）
        print("使用虚拟数据集（用于演示）")
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
            indices = list(range(num_batches))
            if shuffle:
                import random
                random.shuffle(indices)
            for i in indices:
                start = i * batch_size
                end = start + batch_size
                yield data[start:end], targets[start:end]
        
        train_loader = create_loader(train_data, train_targets, config['batch_size'], shuffle=True)
        val_loader = create_loader(val_data, val_targets, config['batch_size'], shuffle=False)
        
        print(f"✓ 训练集: {len(train_data)} 样本")
        print(f"✓ 验证集: {len(val_data)} 样本")
    print()
    
    # 创建学习率调度器（在数据集创建后，以便使用正确的样本数）
    lr_scheduler = None
    if config.get('use_lr_scheduling', True):
        # 计算总步数（使用实际的样本数）
        if use_hf_dataset:
            num_batches_per_epoch = len(train_loader)
        else:
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
        
        # 如果从checkpoint恢复，需要调整学习率调度器的当前步数
        if start_epoch > 0:
            # 计算已完成的步数，并设置调度器的当前步数
            completed_steps = start_epoch * num_batches_per_epoch
            # 设置current_step为已完成的步数（step()会在调用时+1，所以这里不减1）
            lr_scheduler.current_step = completed_steps
            # 手动计算并设置当前学习率（不使用step()避免步数+1）
            if completed_steps <= warmup_steps:
                lr = config['learning_rate'] * (completed_steps / warmup_steps)
            else:
                progress = (completed_steps - warmup_steps) / max(total_steps - warmup_steps, 1)
                lr = config.get('min_lr', 1e-6) + (config['learning_rate'] - config.get('min_lr', 1e-6)) * \
                     (1 + math.cos(math.pi * progress)) / 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"✓ 学习率调度器已恢复 (从步数 {completed_steps} 继续, 当前学习率: {lr:.2e}, 总步数: {total_steps})")
        else:
            print(f"✓ 学习率调度器已启用 (Warmup: {warmup_steps}步, 总步数: {total_steps})")
        print()
    
    # 训练循环
    print("开始训练...")
    print("=" * 80)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start_time = time.time()
        
        # 如果不是使用HuggingFace数据集，需要重新创建数据加载器
        if not use_hf_dataset:
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
            if use_hf_dataset:
                num_batches_per_epoch = len(train_loader)
            else:
                num_batches_per_epoch = config['num_train_samples'] // config['batch_size']
            current_step = epoch * num_batches_per_epoch + num_batches_per_epoch
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
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }
            # 保存scaler状态（如果使用混合精度训练）
            if scaler is not None:
                checkpoint_data['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint_data, 'best_model.pth')
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
