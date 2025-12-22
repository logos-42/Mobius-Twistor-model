"""
快速测试脚本 - 使用D盘上的WikiText-103数据集（无需HuggingFace）
直接读取原始文本文件

使用方法:
  1. 激活虚拟环境: .\\venv_gpu\\Scripts\\activate (PowerShell) 或 venv_gpu\\Scripts\\activate.bat (CMD)
  2. 运行脚本: python examples/quick_test_wikitext103.py
  
  或者直接使用虚拟环境的Python:
  .\\venv_gpu\\Scripts\\python.exe examples/quick_test_wikitext103.py
"""

import torch
import torch.nn as nn
import sys
import os
import re
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mt_transformer import TwistorHopeArchitecture

print("=" * 80)
print("WikiText-103 快速测试（无需HuggingFace）")
print("=" * 80)

# 详细的GPU检测和设置
print("\n检测GPU设备...")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_count = torch.cuda.device_count()
    print(f"✓ 检测到 {gpu_count} 个GPU设备")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_props = torch.cuda.get_device_properties(i)
        gpu_memory_gb = gpu_props.total_memory / (1024**3)
        print(f"  GPU {i}: {gpu_name}")
        print(f"    显存: {gpu_memory_gb:.2f} GB")
        print(f"    计算能力: {gpu_props.major}.{gpu_props.minor}")
    
    # 使用第一个GPU
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    print(f"\n✓ 使用设备: {device}")
    print(f"✓ 当前GPU: {torch.cuda.get_device_name(0)}")
    
    # 测试GPU是否真的可用
    try:
        test_tensor = torch.zeros(1, device=device)
        print(f"✓ GPU测试成功，可以正常使用\n")
    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        print("⚠️  将回退到CPU模式")
        device = torch.device('cpu')
else:
    print("❌ 未检测到GPU！")
    print("\n可能的原因：")
    print("  1. 未安装CUDA版本的PyTorch")
    print("  2. GPU驱动未正确安装")
    print("  3. CUDA版本不匹配")
    print("\n建议：")
    print("  1. 检查是否安装了CUDA版本的PyTorch:")
    print("     - 访问 https://pytorch.org/ 获取正确的安装命令")
    print("     - 例如: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("  2. 检查GPU驱动: nvidia-smi")
    print("  3. 检查CUDA版本: nvcc --version")
    print("\n⚠️  将使用CPU运行（训练会很慢）")
    device = torch.device('cpu')
    print(f"使用设备: {device}\n")

# WikiText-103数据集路径（D盘）
# HuggingFace缓存目录
HF_CACHE_DIR = r"D:\AI\huggingface_cache"

# 尝试在HuggingFace缓存目录中查找WikiText-103数据集
def find_wikitext_dataset(cache_dir):
    """在HuggingFace缓存目录中查找WikiText-103数据集"""
    if not os.path.exists(cache_dir):
        return None
    
    # HuggingFace数据集通常在这个路径下
    possible_paths = [
        # 直接路径
        os.path.join(cache_dir, "wikitext-103"),
        os.path.join(cache_dir, "WikiText-103"),
        # HuggingFace datasets格式
        os.path.join(cache_dir, "datasets", "wikitext", "wikitext-103-raw-v1"),
        os.path.join(cache_dir, "datasets", "wikitext", "wikitext-103-v1"),
        # 其他可能的位置
        os.path.join(cache_dir, "wikitext"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # 递归搜索包含wiki.train.tokens的目录
    print("  递归搜索数据集文件...")
    for root, dirs, files in os.walk(cache_dir):
        # 限制搜索深度，避免搜索太深
        depth = root[len(cache_dir):].count(os.sep)
        if depth > 3:  # 最多搜索3层
            continue
        
        # 检查是否包含wiki.train.tokens文件
        if "wiki.train.tokens" in files or "wiki.train.raw" in files:
            return root
    
    return None

# 查找数据集路径
DATASET_PATH = find_wikitext_dataset(HF_CACHE_DIR)

if DATASET_PATH is None:
    print(f"⚠️  警告: 在 {HF_CACHE_DIR} 中未找到WikiText-103数据集")
    print("尝试其他常见路径...")
    
    # 尝试其他常见路径
    possible_paths = [
        r"D:\wikitext-103",
        r"D:\WikiText-103",
        r"D:\wikitext-103-raw-v1",
        r"D:\wikitext",
        r"D:\WikiText",
    ]
    
    found = False
    for path in possible_paths:
        if os.path.exists(path):
            DATASET_PATH = path
            found = True
            print(f"✓ 找到数据集路径: {DATASET_PATH}")
            break
    
    if not found:
        print("❌ 未找到数据集")
        print(f"\n请确保数据集在以下位置之一:")
        print(f"  - {HF_CACHE_DIR}\\datasets\\wikitext\\wikitext-103-raw-v1")
        print(f"  - {HF_CACHE_DIR}\\wikitext-103")
        print(f"  - D:\\wikitext-103")
        print("\n或者修改脚本中的 HF_CACHE_DIR 变量")
        sys.exit(1)
else:
    print(f"✓ 找到数据集路径: {DATASET_PATH}")

print(f"数据集路径: {DATASET_PATH}\n")

# 查找训练文件
train_file = None
valid_file = None
test_file = None

# 检查常见的文件名
possible_train_files = [
    "wiki.train.tokens",
    "wiki.train.raw",
    "train.txt",
    "train.tokens",
    "wiki.train",
]

possible_valid_files = [
    "wiki.valid.tokens",
    "wiki.valid.raw",
    "valid.txt",
    "valid.tokens",
    "wiki.valid",
]

possible_test_files = [
    "wiki.test.tokens",
    "wiki.test.raw",
    "test.txt",
    "test.tokens",
    "wiki.test",
]

# 查找文件（支持递归搜索）
def find_file_in_dir(directory, filenames, max_depth=2):
    """在目录中查找文件，支持递归搜索"""
    # 首先在当前目录查找
    for filename in filenames:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            return filepath
    
    # 递归搜索子目录（限制深度）
    for root, dirs, files in os.walk(directory):
        depth = root[len(directory):].count(os.sep)
        if depth > max_depth:
            continue
        
        for filename in filenames:
            filepath = os.path.join(root, filename)
            if os.path.exists(filepath):
                return filepath
    
    return None

# 查找文件
if os.path.isdir(DATASET_PATH):
    train_file = find_file_in_dir(DATASET_PATH, possible_train_files)
    valid_file = find_file_in_dir(DATASET_PATH, possible_valid_files)
    test_file = find_file_in_dir(DATASET_PATH, possible_test_files)
else:
    # 如果路径是文件本身
    if "train" in DATASET_PATH.lower():
        train_file = DATASET_PATH
    elif "valid" in DATASET_PATH.lower() or "val" in DATASET_PATH.lower():
        valid_file = DATASET_PATH
    elif "test" in DATASET_PATH.lower():
        test_file = DATASET_PATH

if train_file is None:
    print("⚠️  未找到训练文件")
    print(f"  在目录 {DATASET_PATH} 中查找以下文件:")
    for f in possible_train_files:
        print(f"    - {f}")
    print("\n将使用虚拟数据集进行快速GPU测试...")
    use_dummy_data = True
else:
    use_dummy_data = False

print(f"✓ 找到训练文件: {train_file}")
if valid_file:
    print(f"✓ 找到验证文件: {valid_file}")
if test_file:
    print(f"✓ 找到测试文件: {test_file}")

# 简单的tokenizer（基于空格和标点）
class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"
        self.special_tokens = [self.pad_token, self.unk_token, self.eos_token]
        self.vocab_size = 0
        
    def build_vocab(self, texts, min_freq=2, max_vocab=10000):
        """从文本构建词汇表"""
        print("\n构建词汇表...")
        word_counts = Counter()
        
        for text in texts:
            # 简单的tokenization：按空格和标点分割
            words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
            word_counts.update(words)
        
        # 选择最常见的词
        most_common = word_counts.most_common(max_vocab - len(self.special_tokens))
        
        # 添加特殊token
        self.word_to_id = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.id_to_word = {idx: token for idx, token in enumerate(self.special_tokens)}
        
        # 添加词汇（过滤低频词）
        current_id = len(self.special_tokens)
        for word, count in most_common:
            if count >= min_freq:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
        
        self.vocab_size = len(self.word_to_id)
        print(f"  ✓ 词汇表大小: {self.vocab_size:,}")
        print(f"  ✓ 特殊token: {self.special_tokens}")
        
    def encode(self, text, max_length=None, add_special_tokens=True):
        """将文本编码为token ID列表"""
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.word_to_id.get(self.eos_token, 0))
        
        for word in words:
            token_id = self.word_to_id.get(word, self.word_to_id.get(self.unk_token, 0))
            token_ids.append(token_id)
        
        if max_length:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                # 填充
                pad_id = self.word_to_id.get(self.pad_token, 0)
                token_ids.extend([pad_id] * (max_length - len(token_ids)))
        
        return token_ids

# 读取文本文件
def read_text_file(filepath, max_lines=None):
    """读取文本文件，返回文本列表"""
    texts = []
    print(f"  读取文件: {os.path.basename(filepath)}")
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                line = line.strip()
                if line and len(line) > 10:  # 过滤空行和太短的行
                    texts.append(line)
        print(f"  ✓ 读取了 {len(texts):,} 行")
    except Exception as e:
        print(f"  ❌ 读取文件失败: {e}")
        return []
    
    return texts

# 加载训练数据
print("\n加载训练数据...")
if use_dummy_data:
    # 使用虚拟数据
    print("  使用虚拟数据集进行GPU测试...")
    # 生成一些虚拟文本数据
    dummy_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning models use neural networks with multiple layers.",
        "Natural language processing helps computers understand human language.",
        "Transformers are a type of neural network architecture.",
        "Attention mechanisms allow models to focus on relevant parts of input.",
        "Training neural networks requires large amounts of data.",
        "Gradient descent is used to optimize model parameters.",
        "Backpropagation calculates gradients for neural network training.",
        "Convolutional neural networks are effective for image recognition.",
    ] * 10  # 重复10次以生成更多样本
    train_texts = dummy_texts
    print(f"  ✓ 生成了 {len(train_texts)} 个虚拟样本")
else:
    max_train_lines = 1000  # 快速测试只读取1000行
    train_texts = read_text_file(train_file, max_lines=max_train_lines)
    
    if len(train_texts) == 0:
        print("❌ 没有读取到训练数据")
        sys.exit(1)

# 构建词汇表
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(train_texts, min_freq=1, max_vocab=5000)  # 快速测试使用较小的词汇表

# 创建小模型（快速测试）
print("\n创建模型...")
model = TwistorHopeArchitecture(
    vocab_size=tokenizer.vocab_size,
    dim=128,  # 较小的维度用于快速测试
    hidden_dim=128,
    num_recurrent_layers=2,
    num_memories=4,
    use_nested_learning=True,
    bidirectional=False,
    dropout=0.1
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"✓ 模型参数量: {num_params:,} ({num_params/1e6:.2f}M)")

# 创建优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 准备数据（只使用少量样本进行快速测试）
print("\n准备数据...")
seq_len = 32
batch_size = 4
max_samples = 50  # 只使用50个样本进行快速测试

# 处理数据
print(f"  处理前 {min(max_samples, len(train_texts))} 个样本...")
processed_samples = []
for i, text in enumerate(train_texts):
    if i >= max_samples:
        break
    
    # 分词
    tokens = tokenizer.encode(
        text,
        max_length=seq_len + 1,
        add_special_tokens=True
    )
    
    if len(tokens) >= 2:
        processed_samples.append(tokens)

print(f"✓ 有效样本数: {len(processed_samples)}")

if len(processed_samples) == 0:
    print("❌ 没有有效的训练样本")
    sys.exit(1)

# 创建数据加载器
def create_batch(samples, batch_size):
    """创建批次数据"""
    batches = []
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i+batch_size]
        batch_data = []
        batch_targets = []
        
        for sample in batch_samples:
            # 输入是前seq_len个token，目标是后seq_len个token（右移1位）
            data = sample[:seq_len]
            if len(sample) > seq_len:
                targets = sample[1:seq_len+1]
            else:
                targets = sample[1:] + [tokenizer.word_to_id.get(tokenizer.pad_token, 0)]
            
            # 确保长度正确
            while len(data) < seq_len:
                data.append(tokenizer.word_to_id.get(tokenizer.pad_token, 0))
            while len(targets) < seq_len:
                targets.append(tokenizer.word_to_id.get(tokenizer.pad_token, 0))
            
            batch_data.append(data[:seq_len])
            batch_targets.append(targets[:seq_len])
        
        batches.append((
            torch.tensor(batch_data, dtype=torch.long, device=device),
            torch.tensor(batch_targets, dtype=torch.long, device=device)
        ))
    
    return batches

batches = create_batch(processed_samples, batch_size)
num_batches = len(batches)
print(f"  批次数量: {num_batches}")

# 开始训练测试
print(f"\n开始快速训练测试 ({num_batches} 个batch)...")
print("-" * 80)

# 确认设备和模型位置
print(f"训练设备: {device}")
if device.type == 'cuda':
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    # 检查模型参数是否在GPU上
    next_param = next(model.parameters())
    if next_param.is_cuda:
        print(f"✓ 模型已在GPU上")
    else:
        print(f"⚠️  警告: 模型参数不在GPU上，正在移动到GPU...")
        model = model.to(device)
else:
    print("⚠️  使用CPU训练（速度较慢）")

print()

model.train()
total_loss = 0.0

for i, (token_ids, targets) in enumerate(batches):
    # 确保数据在正确的设备上
    if token_ids.device != device:
        token_ids = token_ids.to(device)
    if targets.device != device:
        targets = targets.to(device)
    
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

print("\n✓ 快速测试完成！")
print("\n提示:")
print("  - 此脚本使用简单的tokenization，不需要HuggingFace库")
print("  - 如需完整训练，可以:")
print("    1. 安装HuggingFace: pip install datasets transformers")
print("    2. 使用: python examples/train_gpu.py --dataset wikitext --dataset-config wikitext-103-raw-v1")
print("    3. 或修改此脚本，增加 max_samples 和 epochs 进行完整训练")
print("=" * 80)
