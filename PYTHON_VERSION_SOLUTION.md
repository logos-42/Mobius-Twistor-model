# Python版本兼容性解决方案

## 问题说明

您当前使用的是 **Python 3.14**，这是一个非常新的版本。PyTorch目前还没有为Python 3.14提供CUDA版本的预编译包，只有CPU版本可用。

## 解决方案

### 方案1：使用Python 3.11或3.12虚拟环境（推荐）⭐⭐⭐

这是最佳解决方案，可以获得完整的GPU支持。

#### 步骤：

1. **下载Python 3.12**
   - 访问 https://www.python.org/downloads/
   - 下载Python 3.12.x（推荐3.12.x，稳定且PyTorch完全支持）

2. **安装Python 3.12后，创建虚拟环境：**
   ```bash
   # 使用Python 3.12创建虚拟环境
   py -3.12 -m venv venv_gpu
   
   # 激活虚拟环境（Windows PowerShell）
   .\venv_gpu\Scripts\Activate.ps1
   
   # 或者（Windows CMD）
   venv_gpu\Scripts\activate.bat
   ```

3. **在虚拟环境中安装CUDA版PyTorch：**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **验证安装：**
   ```bash
   python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"
   ```

5. **安装其他依赖：**
   ```bash
   pip install -r requirements.txt
   ```

6. **运行训练：**
   ```bash
   python examples/train_gpu.py
   ```

---

### 方案2：使用CPU版本训练（临时方案）⚠️

如果暂时无法切换Python版本，可以在当前Python 3.14环境下使用CPU版本训练，但速度会很慢。

#### 优点：
- 无需切换Python版本
- 可以立即开始训练

#### 缺点：
- 训练速度慢（比GPU慢10-50倍）
- 不适合大型模型训练

#### 使用：
```bash
# 当前已经安装了CPU版本的PyTorch
python examples/train_gpu.py
```

脚本会自动检测到CPU并相应调整配置。

---

### 方案3：使用Conda环境（如果已安装Conda）

如果您已经安装了Anaconda或Miniconda：

```bash
# 创建Python 3.11环境
conda create -n pytorch_gpu python=3.11
conda activate pytorch_gpu

# 安装CUDA版PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 验证
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 推荐配置

对于GTX 1650 Ti (4GB显存)，推荐使用：
- **Python 3.11 或 3.12**
- **PyTorch 2.x with CUDA 12.1**
- **Python 3.14**：仅用于开发，不用于深度学习训练（等待PyTorch支持）

## 快速检查

运行以下命令检查当前Python版本：
```bash
python --version
```

如果是Python 3.14，建议切换到Python 3.11或3.12以获得GPU支持。

## 需要帮助？

如果遇到问题：
1. 检查nvidia-smi确认GPU驱动正常
2. 确认安装了正确的Python版本
3. 确保虚拟环境已激活
4. 重新安装PyTorch CUDA版本
