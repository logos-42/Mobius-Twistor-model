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

### HOPE架构集成

```python
from mt_transformer import HopeIntegration

# 创建完整的HOPE-MT模型
model = HopeIntegration(
    vocab_size=1000,
    dim=128,
    num_titan_layers=2,
    use_nested_optimizer=False
)

# 前向传播
token_ids = torch.randint(0, 1000, (2, 10))
output = model(token_ids)  # (batch, seq, dim)
```

## 运行测试

```bash
python examples/demo.py
```

## 架构流程

```
Token IDs 
  → SpinorEmbedding (扭量表示: ω和π)
  → Self-Modifying Titans (包含IncidenceAttention)
  → MobiusLayer (拓扑变换)
  → Continuum Memory System (记忆更新)
  → 输出
```

## 关键特性

1. **复数扭量表示** - 每个Token用两个复数分量（ω, π）表示
2. **关联注意力** - 基于扭量关联度，包含莫比乌斯项（缠绕数）
3. **拓扑循环** - 莫比乌斯层实现手性翻转和复共轭
4. **嵌套学习** - 支持嵌套优化范式
5. **持续学习** - 与HOPE架构的CMS集成，支持持续学习

## 文件结构

```
mobi model/
├── model.md                    # 理论文档
├── NL.pdf                      # HOPE架构论文
├── requirements.txt            # 依赖包列表
├── mt_transformer/
│   ├── __init__.py            # 包初始化
│   ├── spinor_embedding.py    # 旋量嵌入层
│   ├── incidence_attention.py # 关联注意力机制
│   ├── mobius_layer.py        # 莫比乌斯层
│   ├── nested_optimizer.py   # 嵌套优化器包装
│   └── hope_integration.py    # HOPE架构集成
├── examples/
│   └── demo.py                # 使用示例和测试
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

