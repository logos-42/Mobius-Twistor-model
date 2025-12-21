# Python 3.12 GPU环境设置脚本
# 用于安装Python 3.12并配置CUDA版PyTorch环境

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Python 3.12 GPU环境自动设置脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查Python 3.12是否已安装
Write-Host "[1/5] 检查Python 3.12..." -ForegroundColor Yellow
try {
    $python312 = py -3.12 --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Python 3.12 已安装: $python312" -ForegroundColor Green
        $pythonInstalled = $true
    } else {
        throw "Python 3.12 not found"
    }
} catch {
    Write-Host "✗ Python 3.12 未安装" -ForegroundColor Red
    Write-Host ""
    Write-Host "尝试自动安装Python 3.12..." -ForegroundColor Yellow
    
    # 尝试使用winget安装
    $wingetCheck = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCheck) {
        Write-Host "使用winget安装Python 3.12..." -ForegroundColor Yellow
        winget install Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
        Start-Sleep -Seconds 5
        # 重新检查
        try {
            $python312 = py -3.12 --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✓ Python 3.12 安装成功: $python312" -ForegroundColor Green
                $pythonInstalled = $true
            }
        } catch {
            $pythonInstalled = $false
        }
    }
    
    if (-not $pythonInstalled) {
        Write-Host "自动安装失败，请手动安装Python 3.12:" -ForegroundColor Yellow
        Write-Host "1. 访问: https://www.python.org/downloads/" -ForegroundColor White
        Write-Host "2. 下载 Python 3.12.x (64-bit)" -ForegroundColor White
        Write-Host "3. 安装时务必勾选 'Add Python to PATH'" -ForegroundColor White
        Write-Host "4. 重新运行此脚本" -ForegroundColor White
        Write-Host ""
        Write-Host "或者运行以下命令（如果已安装winget）:" -ForegroundColor Cyan
        Write-Host "  winget install Python.Python.3.12" -ForegroundColor White
        Write-Host ""
        $pythonInstalled = $false
    }
}

if ($pythonInstalled) {
    # 创建虚拟环境
    Write-Host ""
    Write-Host "[2/5] 创建Python 3.12虚拟环境..." -ForegroundColor Yellow
    if (Test-Path "venv_gpu") {
        Write-Host "虚拟环境已存在，删除旧环境..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force "venv_gpu"
    }
    
    py -3.12 -m venv venv_gpu
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ 虚拟环境创建成功" -ForegroundColor Green
    } else {
        Write-Host "✗ 虚拟环境创建失败" -ForegroundColor Red
        exit 1
    }
    
    # 激活虚拟环境并安装依赖
    Write-Host ""
    Write-Host "[3/5] 激活虚拟环境并升级pip..." -ForegroundColor Yellow
    & ".\venv_gpu\Scripts\python.exe" -m pip install --upgrade pip
    Write-Host "✓ pip 升级完成" -ForegroundColor Green
    
    # 安装CUDA版PyTorch
    Write-Host ""
    Write-Host "[4/5] 安装CUDA版PyTorch (CUDA 12.1)..." -ForegroundColor Yellow
    Write-Host "这可能需要几分钟，请耐心等待..." -ForegroundColor White
    & ".\venv_gpu\Scripts\pip.exe" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ PyTorch CUDA版本安装成功" -ForegroundColor Green
    } else {
        Write-Host "✗ PyTorch安装失败" -ForegroundColor Red
        exit 1
    }
    
    # 安装其他依赖
    Write-Host ""
    Write-Host "[5/5] 安装其他依赖..." -ForegroundColor Yellow
    & ".\venv_gpu\Scripts\pip.exe" install -r requirements.txt
    Write-Host "✓ 依赖安装完成" -ForegroundColor Green
    
    # 验证安装
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "验证GPU支持..." -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    $verifyScript = @"
import torch
import sys

print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print("✓ GPU环境配置成功！" + " " * 20)
else:
    print("✗ CUDA不可用，请检查安装")
    sys.exit(1)
"@
    
    $verifyScript | & ".\venv_gpu\Scripts\python.exe"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "✓ 环境配置完成！" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "激活虚拟环境:" -ForegroundColor Yellow
        Write-Host "  .\venv_gpu\Scripts\Activate.ps1" -ForegroundColor White
        Write-Host ""
        Write-Host "运行GPU训练:" -ForegroundColor Yellow
        Write-Host "  python examples/train_gpu.py" -ForegroundColor White
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "✗ GPU验证失败" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "按任意键退出..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

