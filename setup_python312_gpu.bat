@echo off
REM Python 3.12 GPU环境设置脚本 (CMD版本)
REM 用于安装Python 3.12并配置CUDA版PyTorch环境

echo ========================================
echo Python 3.12 GPU环境自动设置脚本
echo ========================================
echo.

REM 检查Python 3.12是否已安装
echo [1/5] 检查Python 3.12...
py -3.12 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Python 3.12 已安装
    set PYTHON_INSTALLED=1
) else (
    echo Python 3.12 未安装
    echo.
    echo 请手动安装Python 3.12:
    echo 1. 访问: https://www.python.org/downloads/
    echo 2. 下载 Python 3.12.x (64-bit)
    echo 3. 安装时务必勾选 'Add Python to PATH'
    echo 4. 重新运行此脚本
    echo.
    pause
    exit /b 1
)

if %PYTHON_INSTALLED% EQU 1 (
    REM 创建虚拟环境
    echo.
    echo [2/5] 创建Python 3.12虚拟环境...
    if exist venv_gpu (
        echo 虚拟环境已存在，删除旧环境...
        rmdir /s /q venv_gpu
    )
    
    py -3.12 -m venv venv_gpu
    if %ERRORLEVEL% NEQ 0 (
        echo 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo 虚拟环境创建成功
    
    REM 升级pip
    echo.
    echo [3/5] 激活虚拟环境并升级pip...
    call venv_gpu\Scripts\activate.bat
    python -m pip install --upgrade pip
    
    REM 安装CUDA版PyTorch
    echo.
    echo [4/5] 安装CUDA版PyTorch (CUDA 12.1)...
    echo 这可能需要几分钟，请耐心等待...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if %ERRORLEVEL% NEQ 0 (
        echo PyTorch安装失败
        pause
        exit /b 1
    )
    echo PyTorch CUDA版本安装成功
    
    REM 安装其他依赖
    echo.
    echo [5/5] 安装其他依赖...
    pip install -r requirements.txt
    echo 依赖安装完成
    
    REM 验证安装
    echo.
    echo ========================================
    echo 验证GPU支持...
    echo ========================================
    python -c "import torch; import sys; print(f'Python版本: {sys.version}'); print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
    
    echo.
    echo ========================================
    echo 环境配置完成！
    echo ========================================
    echo.
    echo 激活虚拟环境:
    echo   venv_gpu\Scripts\activate.bat
    echo.
    echo 运行GPU训练:
    echo   python examples/train_gpu.py
    echo.
)

pause

