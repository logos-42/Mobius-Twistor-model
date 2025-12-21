# GPU训练环境配置指南

## 当前状态

您的当前环境安装了PyTorch CPU版本，无法使用GPU。要在GTX 1650上训练，需要安装CUDA版本的PyTorch。

## 安装CUDA版本的PyTorch

### 步骤1：检查CUDA版本

首先检查您的系统是否已安装CUDA：

```bash
nvcc --version
```

如果没有安装CUDA，请先安装：
- **GTX 1650** 支持 **CUDA 10.0 - CUDA 12.x**
- 推荐安装 **CUDA 11.8** 或 **CUDA 12.1**

### 步骤2：安装PyTorch（CUDA版本）

访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 获取正确的安装命令。

**对于CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**对于CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 步骤3：验证GPU支持

运行以下命令验证GPU是否可用：

```bash
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"
```

应该看到：
```
CUDA可用: True
GPU: NVIDIA GeForce GTX 1650
```

## 训练脚本使用

### 1. 性能测试

```bash
python examples/performance_test.py
```

这将自动检测GPU并显示详细的性能报告和优化建议。

### 2. 开始训练

```bash
python examples/train_gpu.py
```

训练脚本特性：
- ✅ 自动GPU检测
- ✅ 混合精度训练（节省50%显存）
- ✅ 梯度累积（增大有效batch size）
- ✅ 显存监控
- ✅ 自动保存最佳模型

### 3. 自定义配置

编辑 `examples/train_gpu.py` 中的 `config` 字典来调整：
- `batch_size`: 批次大小（默认8，如果显存不足可减小）
- `dim`: 模型维度（默认128）
- `num_recurrent_layers`: 循环层数（默认2）
- `gradient_accumulation_steps`: 梯度累积步数（默认2）
- `use_amp`: 混合精度训练（默认True）

## GTX 1650 (4GB显存) 优化建议

### 推荐的模型配置：

**小模型（显存占用约1-2GB）：**
```python
config = {
    'dim': 128,
    'num_recurrent_layers': 2,
    'batch_size': 8,
    'seq_len': 32,
}
```

**中模型（显存占用约2-3GB）：**
```python
config = {
    'dim': 128,
    'num_recurrent_layers': 2,
    'batch_size': 4,  # 减小batch size
    'seq_len': 64,
    'gradient_accumulation_steps': 4,  # 通过梯度累积保持有效batch size
}
```

**大模型（显存占用约3-4GB）：**
```python
config = {
    'dim': 256,
    'num_recurrent_layers': 3,
    'batch_size': 2,
    'seq_len': 32,
    'gradient_accumulation_steps': 8,  # 有效batch size = 16
    'use_amp': True,  # 必须启用混合精度
}
```

### 如果遇到显存不足（OOM）：

1. **减小batch_size**（最有效）
2. **启用混合精度训练**（`use_amp=True`）
3. **增加梯度累积步数**（保持训练效果）
4. **减小模型维度**（`dim`）
5. **减小序列长度**（`seq_len`）
6. **定期清理显存**（脚本已自动处理）

## 监控GPU使用情况

### Windows任务管理器
打开任务管理器 → 性能 → GPU，可以查看显存使用情况。

### 命令行监控（如果安装了nvidia-smi）
```bash
nvidia-smi -l 1  # 每秒刷新一次
```

## 训练输出说明

训练脚本会显示：
- 每个epoch的训练/验证损失
- GPU显存使用情况
- 自动保存最佳模型到 `best_model.pth`

## 加载保存的模型

```python
import torch
from mt_transformer import TwistorHopeArchitecture

checkpoint = torch.load('best_model.pth')
config = checkpoint['config']

model = TwistorHopeArchitecture(**config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 故障排除

### 问题1：CUDA不可用
- 确认安装了CUDA版本的PyTorch（不是CPU版本）
- 检查CUDA驱动是否正确安装

### 问题2：显存不足（OOM）
- 减小batch_size
- 启用混合精度训练
- 减小模型规模

### 问题3：训练速度慢
- 确保数据在GPU上（脚本已自动处理）
- 检查是否有其他程序占用GPU
- 考虑使用更大的batch_size（在显存允许的情况下）

## 需要帮助？

如果遇到问题，检查：
1. PyTorch版本和CUDA版本是否匹配
2. GPU驱动是否最新
3. 显存是否被其他程序占用
