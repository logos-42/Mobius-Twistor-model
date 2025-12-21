# Python 3.12 安装和GPU环境配置完整指南

## 为什么需要Python 3.12？

- ✅ Python 3.14太新，PyTorch还没有CUDA版本支持
- ✅ Python 3.12是PyTorch官方完全支持的稳定版本
- ✅ 性能优秀，兼容性好

## 第一步：下载并安装Python 3.12

### 方法1：官方网站下载（推荐）

1. **访问Python官网**
   - 打开：https://www.python.org/downloads/
   - 或直接下载：https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe

2. **安装Python 3.12**
   - 双击下载的安装程序
   - ⚠️ **重要：勾选 "Add Python 3.12 to PATH"**
   - 选择 "Install Now" 或 "Customize installation"
   - 点击 "Install"

3. **验证安装**
   - 打开新的PowerShell窗口
   - 运行：`py -3.12 --version`
   - 应该看到：`Python 3.12.x`

### 方法2：使用Microsoft Store

1. 打开Microsoft Store
2. 搜索 "Python 3.12"
3. 安装 "Python 3.12"（由Python Software Foundation发布）

## 第二步：运行自动设置脚本

### 使用PowerShell（推荐）

```powershell
# 在项目目录下运行
.\setup_python312_gpu.ps1
```

### 使用CMD

```cmd
# 在项目目录下运行
setup_python312_gpu.bat
```

### 手动设置（如果脚本失败）

```powershell
# 1. 创建虚拟环境
py -3.12 -m venv venv_gpu

# 2. 激活虚拟环境
.\venv_gpu\Scripts\Activate.ps1

# 3. 升级pip
python -m pip install --upgrade pip

# 4. 安装CUDA版PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. 安装其他依赖
pip install -r requirements.txt

# 6. 验证安装
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"
```

## 第三步：验证GPU支持

```powershell
# 激活虚拟环境
.\venv_gpu\Scripts\Activate.ps1

# 运行验证
python -c "import torch; print('✓ PyTorch版本:', torch.__version__); print('✓ CUDA可用:', torch.cuda.is_available()); print('✓ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"
```

应该看到：
```
✓ PyTorch版本: 2.x.x+cu121
✓ CUDA可用: True
✓ GPU: NVIDIA GeForce GTX 1650 Ti
```

## 第四步：运行GPU训练

```powershell
# 确保虚拟环境已激活
.\venv_gpu\Scripts\Activate.ps1

# 运行性能测试
python examples/performance_test.py

# 运行GPU训练
python examples/train_gpu.py
```

## 常见问题

### 问题1：`py -3.12` 命令找不到

**解决：**
- 确认Python 3.12已安装
- 确认安装时勾选了 "Add Python to PATH"
- 重新打开PowerShell窗口
- 或使用完整路径：`C:\Users\YourName\AppData\Local\Programs\Python\Python312\python.exe`

### 问题2：PowerShell脚本执行策略限制

**解决：**
```powershell
# 临时允许脚本执行（仅当前会话）
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# 然后运行脚本
.\setup_python312_gpu.ps1
```

### 问题3：CUDA仍然不可用

**检查清单：**
1. ✅ 确认nvidia-smi可以显示GPU
2. ✅ 确认安装了CUDA版本的PyTorch（不是CPU版本）
3. ✅ 确认虚拟环境已激活
4. ✅ 尝试重新安装：`pip uninstall torch torchvision torchaudio -y` 然后重新安装

### 问题4：安装速度慢

**解决：**
- PyTorch包很大（~2GB），需要耐心等待
- 可以使用国内镜像加速（如果在中国）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 项目结构

安装完成后，项目结构如下：

```
mobi model/
├── venv_gpu/              # Python 3.12虚拟环境
│   ├── Scripts/
│   │   └── activate.ps1   # 激活脚本
│   └── ...
├── examples/
│   ├── performance_test.py
│   └── train_gpu.py
├── setup_python312_gpu.ps1
├── setup_python312_gpu.bat
└── ...
```

## 快速参考

### 每次使用GPU训练：

```powershell
# 1. 激活虚拟环境
cd "d:\AI\model\mobi model"
.\venv_gpu\Scripts\Activate.ps1

# 2. 运行训练
python examples/train_gpu.py
```

### 退出虚拟环境：

```powershell
deactivate
```

## 需要帮助？

如果遇到问题，检查：
1. Python 3.12是否正确安装
2. 虚拟环境是否正确创建
3. PyTorch是否为CUDA版本（不是+cpu）
4. GPU驱动是否正常（nvidia-smi）

