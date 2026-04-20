# hw2-setup.ps1
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  cv-intro hw2 env setup (PowerShell)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check if Conda is installed
Write-Host "[1/6] Checking Conda installation status..." -ForegroundColor Yellow
try {
    $condaVersion = conda --version 2>&1
    Write-Host "[OK] Conda is installed: $condaVersion" -ForegroundColor Green
} catch {
    Write-Host "[Err] Conda not found. Please install Anaconda or Miniconda first" -ForegroundColor Red
    Write-Host "Download: https://www.anaconda.com/download" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# 2. Create Conda environment
$ENV_NAME = "cv-hw2"
$PYTHON_VERSION = "3.9"

Write-Host "[2/6] Creating Conda environment: $ENV_NAME (Python $PYTHON_VERSION)" -ForegroundColor Yellow
conda env remove --name $ENV_NAME -y 2>&1 | Out-Null
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

if ($LASTEXITCODE -ne 0) {
    Write-Host "[Err] Environment creation failed" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[OK] Environment created successfully" -ForegroundColor Green
Write-Host ""

# 3. Configure Tsinghua mirror
Write-Host "[3/6] Configuring Tsinghua mirror..." -ForegroundColor Yellow
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --set show_channel_urls yes
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | Out-Null
Write-Host "[OK] Mirror configuration completed" -ForegroundColor Green
Write-Host ""

# 4. Install PyTorch (need to activate environment first)
Write-Host "[4/6] Installing PyTorch and torchvision (CPU version)..." -ForegroundColor Yellow
Write-Host "Note: This may take a few minutes, please be patient..." -ForegroundColor Gray

# Initialize conda for PowerShell
conda init powershell 2>&1 | Out-Null
. $HOME\Documents\WindowsPowerShell\profile.ps1 2>&1 | Out-Null

# Activate environment
conda activate $ENV_NAME

conda install pytorch torchvision cpuonly -y

if ($LASTEXITCODE -ne 0) {
    Write-Host "[Warning] Conda installation failed, trying pip..." -ForegroundColor Yellow
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}
Write-Host "[OK] PyTorch installation completed" -ForegroundColor Green
Write-Host ""

# 5. Install other dependencies
Write-Host "[5/6] Installing other dependencies..." -ForegroundColor Yellow

Write-Host "  - Installing tqdm..." -ForegroundColor Gray
pip install tqdm -q

Write-Host "  - Installing opencv-python..." -ForegroundColor Gray
pip install opencv-python -q

Write-Host "  - Installing pillow..." -ForegroundColor Gray
pip install pillow -q

Write-Host "  - Installing tensorboardx..." -ForegroundColor Gray
pip install tensorboardx -q

Write-Host "  - Installing tensorflow..." -ForegroundColor Gray
pip install tensorflow -q

# 6. Handle numpy compatibility
Write-Host "  - Handling numpy version compatibility..." -ForegroundColor Gray
$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($pythonVersion -lt "3.9") {
    Write-Host "  [Skip] Python version $pythonVersion (< 3.9), no need to downgrade numpy" -ForegroundColor Gray
} else {
    Write-Host "  [Exec] Python $pythonVersion (>= 3.9), downgrading numpy to 1.26.4" -ForegroundColor Gray
    pip install numpy==1.26.4 -q
}

Write-Host "[OK] All dependencies installed" -ForegroundColor Green
Write-Host ""

# 7. Verify installation
Write-Host "[6/6] Verifying installation..." -ForegroundColor Yellow

$verifyScript = @"
import torch
import torchvision
import cv2
import PIL
import tqdm
print('[OK] All libraries imported successfully')
"@

$verifyResult = python -c $verifyScript 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[Err] Some libraries failed verification" -ForegroundColor Red
    Write-Host $verifyResult -ForegroundColor Red
} else {
    Write-Host "[OK] Environment verification passed" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Environment installation completed!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next step: conda activate $ENV_NAME" -ForegroundColor Yellow
Write-Host ""

Read-Host "Press Enter to exit"