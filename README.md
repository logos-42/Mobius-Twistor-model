# MT-Transformer: Möbius-Twistor Transformer Components

基于莫比乌斯-扭量理论的Transformer核心组件，集成到HOPE架构中。

## 概述

本项目实现了基于扭量理论（Twistor Theory）和莫比乌斯拓扑结构的深度学习组件，并将其集成到Google的HOPE架构（基于Nested Learning范式）中。

### 核心组件

1. **旋量嵌入层 (Spinor Embedding)** - 将Token映射为复数扭量表示（ω和π两个分量）
2. **关联注意力机制 (Incidence Attention)** - 基于扭量关联度的注意力计算，包含莫比乌斯项
3. **莫比乌斯层 (Möbius Layer)** - 实现拓扑循环结构，支持手性翻转和复共轭
4. **嵌套优化器 (Nested Optimizer)** - 支持嵌套学习范式的优化器包装
5. **HOPE架构集成 (Hope Integration)** - 完整的HOPE-MT架构实现

## 理论背景

### 扭量理论 (Twistor Theory)

传统Transformer将Token看作高维空间中的**点（向量）**，而MT-Transformer将Token看作**光线（扭量）**。

- **传统模型**：关注点与点的距离（点积注意力）
- **MT-Transformer**：关注线与线的相交关系（关联关系），并在层与层之间引入莫比乌斯拓扑结构

### HOPE架构

HOPE（Self-modifying Architecture with Continuum Memory）是Google Research提出的基于Nested Learning范式的架构，包含：

- **Self-Modifying Titans** - 自我修正序列模型
- **Continuum Memory System (CMS)** - 连续记忆系统
- **Nested Learning Paradigm** - 嵌套学习范式

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 基本使用

```python
import torch
from mt_transformer import SpinorEmbedding, IncidenceAttention, MobiusLayer

# 创建组件
vocab_size = 1000
dim = 128
embedding = SpinorEmbedding(vocab_size=vocab_size, dim=dim // 2)
attention = IncidenceAttention(dim=dim)
mobius = MobiusLayer(dim=dim)

# 使用
token_ids = torch.randint(0, vocab_size, (2, 10))  # (batch, seq)
x = embedding(token_ids)  # (batch, seq, dim)
x = attention(x)
x = mobius(x)
```

### 扭量化HOPE架构

```python
from mt_transformer import TwistorHopeArchitecture
from mt_transformer.model_configs import create_full_config

# 使用推荐配置
full_config = create_full_config(config_name='recommended', vocab_size=1000)
model = TwistorHopeArchitecture(**full_config)

# 前向传播
token_ids = torch.randint(0, 1000, (2, 32))
output, constraint_loss = model(token_ids, return_constraint_loss=True)
```

### 使用优化训练脚本

```python
# 直接运行优化训练脚本
python examples/train_optimized.py --config recommended

# 或使用Python API
from examples.train_optimized import create_optimized_model

model, config = create_optimized_model(
    config_name='recommended',
    vocab_size=1000,
    device=torch.device('cuda')
)
```

## 运行测试

```bash
python examples/demo.py
```

## 模型配置

项目提供了多种预设配置，适用于不同的计算资源和性能需求：

### 预设配置

```python
from mt_transformer.model_configs import get_config, create_full_config

# 获取预设配置
config = get_config('recommended')  # 'small', 'medium', 'recommended', 'large'

# 创建完整配置（包含所有必需参数）
full_config = create_full_config(
    config_name='recommended',
    vocab_size=1000
)
```

### 配置对比

| 配置 | 维度 | 循环层数 | 嵌套层级 | 记忆数量 | 双向 | 参数量(估算) |
|------|------|----------|----------|----------|------|--------------|
| small | 128 | 2 | 5 | 3 | 否 | ~4.7M |
| medium | 192 | 2 | 5 | 4 | 否 | ~8.5M |
| **recommended** | **256** | **3** | **6** | **5** | **是** | **~25M** |
| large | 512 | 4 | 8 | 8 | 是 | ~100M |

### 推荐配置

推荐配置 (`recommended`) 提供了性能与资源消耗的最佳平衡：

- **维度**: 256
- **循环层数**: 3
- **嵌套层级**: 6
- **记忆数量**: 5
- **双向循环**: 是
- **参数量**: ~25M

适合在GTX 1650 (4GB显存) 等中等GPU上训练，通过混合精度训练可进一步降低显存占用。

## 架构流程

### 扭量化HOPE架构（Recurrent版本）

```
Token IDs 
  → SpinorEmbedding (扭量表示: ω和π)
  → TwistorSelfModifyingRecurrent (扭量自我修正，Recurrent版本)
    → TwistorTitansRecurrent (扭量化循环层)
    → AdaptiveMobiusLayer (自适应莫比乌斯层)
  → TwistorMemorySystem (扭量记忆系统，循环更新)
  → TwistorNestedLearning (扭量嵌套学习)
  → 输出
```

**注意**: 这是真正的Recurrent结构，完全移除了注意力机制，符合Google HOPE架构的Titans设计。

## 关键特性

1. **复数扭量表示** - 每个Token用两个复数分量（ω, π）表示
2. **扭量演化** - 自适应演化率和多尺度演化模式
3. **拓扑循环** - 多层莫比乌斯循环，支持自适应耦合系数
4. **嵌套学习** - 多层级嵌套学习（默认5层，推荐配置6层），支持动态权重和层级学习率
5. **持续学习** - 与HOPE架构的CMS集成，支持持续学习
6. **Recurrent架构** - 真正的循环结构，完全移除注意力机制

## 训练优化

### 混合精度训练 (AMP)

使用混合精度训练可以节省约50%的显存，同时提升约30%的训练速度：

```python
# 在 train_gpu.py 中自动启用（GPU环境）
use_amp = True  # 自动检测GPU并启用
```

### 梯度累积

通过梯度累积可以模拟更大的batch size，而不增加显存占用：

```python
gradient_accumulation_steps = 4  # 等效batch size = batch_size * 4
```

### 学习率调度

支持Warmup + Cosine退火学习率调度：

```python
# 自动配置（train_gpu.py）
use_lr_scheduling = True
warmup_ratio = 0.1  # 前10%步数用于warmup
min_lr = 1e-6       # 最小学习率
```

### 优化训练脚本

使用 `train_optimized.py` 可以快速开始优化训练：

```bash
# 使用推荐配置
python examples/train_optimized.py --config recommended

# 使用小配置（适合4GB显存）
python examples/train_optimized.py --config small

# 自定义参数
python examples/train_optimized.py \
    --config recommended \
    --batch-size 16 \
    --num-epochs 20 \
    --learning-rate 2e-4
```

### 训练优化功能

- ✅ **混合精度训练** - 自动检测GPU并启用，节省50%显存
- ✅ **梯度累积** - 支持自定义累积步数
- ✅ **学习率调度** - Warmup + Cosine退火
- ✅ **NaN检测和跳过** - 自动跳过包含NaN/Inf的batch
- ✅ **梯度裁剪** - 防止梯度爆炸
- ✅ **动态嵌套权重** - 根据训练进度调整嵌套权重
- ✅ **层级学习率** - 深层使用较小学习率，浅层使用较大学习率

## 性能对比

### 不同配置的性能指标

| 配置 | 参数量 | 内存占用 | 前向传播 | 训练速度 | 适用场景 |
|------|--------|----------|----------|----------|----------|
| small | 4.7M | ~36 MB | ~15 ms | 快 | 快速实验、小数据集 |
| medium | 8.5M | ~65 MB | ~22 ms | 中等 | 中等规模任务 |
| **recommended** | **25M** | **~150 MB** | **~35 ms** | **中等** | **推荐配置，最佳平衡** |
| large | 100M | ~600 MB | ~120 ms | 慢 | 大规模任务、高性能GPU |

*注：内存占用为训练时估算值，使用混合精度训练可降低约50%*

### 优化效果

使用优化训练脚本 (`train_optimized.py`) 的效果：

- **显存节省**: 混合精度训练可节省约50%显存
- **训练速度**: 提升约30%（混合精度 + 优化）
- **稳定性**: NaN检测和梯度裁剪提升训练稳定性
- **收敛速度**: 学习率调度和动态权重优化收敛速度

### 性能测试

运行性能测试脚本：

```bash
python examples/performance_test.py
```

测试内容包括：
- 参数量统计
- 内存占用估算
- 前向传播速度
- 训练速度
- 不同配置对比

## 文件结构

```
mobi model/
├── model.md                    # 理论文档
├── NL.pdf                      # HOPE架构论文
├── requirements.txt            # 依赖包列表
├── mt_transformer/
│   ├── __init__.py            # 包初始化
│   ├── model_configs.py       # 模型配置模块
│   ├── spinor_embedding.py    # 旋量嵌入层
│   ├── incidence_attention.py # 关联注意力机制
│   ├── mobius_layer.py        # 莫比乌斯层
│   ├── nested_optimizer.py   # 嵌套优化器包装
│   ├── twistor_titans_cell.py # 扭量Titans单元
│   ├── twistor_titans_recurrent.py # 扭量循环层
│   ├── twistor_self_modifying_recurrent.py # 扭量自我修正（Recurrent）
│   ├── twistor_memory_system.py # 扭量记忆系统
│   ├── twistor_nested_learning.py # 扭量嵌套学习
│   └── twistor_hope_architecture.py # 完整扭量化HOPE架构
├── examples/
│   ├── demo.py                # 基础使用示例
│   ├── train_gpu.py           # GPU训练脚本（增强版）
│   ├── train_optimized.py     # 优化训练脚本（推荐）
│   └── performance_test.py    # 性能测试脚本
└── README.md                   # 使用说明
```

## 技术细节

### 复数表示

默认使用**分离实部虚部**的方式（而非`torch.complex64`），便于与HOPE架构的混合计算兼容。

### 维度说明

- 输入维度 `dim`：每个分量的维度
- 旋量嵌入输出：`dim * 2`（ω和π各占`dim`）
- 其他组件输入/输出：`dim * 2`（扭量表示）

### 数值稳定性

注意复数运算的数值精度和梯度稳定性。建议使用梯度裁剪和适当的初始化。

## 参考文献

1. **Nested Learning: The Illusion of Deep Learning Architectures** - Google Research (NeurIPS 2025)
2. **Twistor Theory** - 扭量理论在深度学习中的应用
3. **Möbius Transformations** - 莫比乌斯变换在神经网络中的应用

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

