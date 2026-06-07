# Setup script for SAM 2 Rotoscoper on Windows with RTX 4060 (CUDA)
$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " Starting SAM 2 Rotoscoper Setup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# 1. Create Virtual Environment
if (-not (Test-Path "venv")) {
    Write-Host "[1/6] Creating Python virtual environment (venv)..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "[1/6] Virtual environment already exists. Skipping creation." -ForegroundColor Green
}

# 2. Upgrade pip
Write-Host "[2/6] Upgrading pip..." -ForegroundColor Yellow
& .\venv\Scripts\python.exe -m pip install --upgrade pip
Write-Host "Pip upgraded successfully." -ForegroundColor Green

# 3. Install PyTorch with CUDA 12.4 support
Write-Host "[3/6] Installing PyTorch with CUDA 12.4 support..." -ForegroundColor Yellow
Write-Host "This will take a few minutes as it downloads large wheel files. Please wait..." -ForegroundColor Gray
& .\venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu124
Write-Host "PyTorch & torchvision with CUDA successfully installed." -ForegroundColor Green

# 4. Initialize and update SAM 2 Git Submodule
if (-not (Test-Path "sam2_repo\setup.py")) {
    Write-Host "[4/6] Initializing and updating SAM 2 Git submodule..." -ForegroundColor Yellow
    git submodule update --init --recursive
    Write-Host "SAM 2 submodule updated successfully." -ForegroundColor Green
} else {
    Write-Host "[4/6] SAM 2 submodule already initialized." -ForegroundColor Green
}

# 5. Install SAM 2 in editable mode (without compiling CUDA custom extensions)
Write-Host "[5/6] Installing SAM 2 in editable mode (bypassing custom CUDA compile)..." -ForegroundColor Yellow
$env:SAM2_BUILD_CUDA = "0"
& .\venv\Scripts\pip.exe install -e .\sam2_repo
Write-Host "SAM 2 library successfully installed in pure PyTorch mode." -ForegroundColor Green

# 6. Install other requirements
Write-Host "[6/6] Installing additional web and image processing requirements..." -ForegroundColor Yellow
& .\venv\Scripts\pip.exe install -r requirements.txt
Write-Host "Additional requirements installed successfully." -ForegroundColor Green

# Create checkpoints directory and download weights
if (-not (Test-Path "checkpoints")) {
    New-Item -ItemType Directory -Path "checkpoints" | Out-Null
}

Write-Host "Downloading SAM 2.1 Hiera Large model weights..." -ForegroundColor Yellow
$ckpt_file = "checkpoints\sam2.1_hiera_large.pt"
if (-not (Test-Path $ckpt_file)) {
    # We download via Python to show nice downloading status or handle it cleanly
    $download_code = @"
import urllib.request
import os
import sys

url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
output = "checkpoints/sam2.1_hiera_large.pt"

print(f"Downloading {url} to {output}...")

def progress_hook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rDownloading... {percent}% ({count * block_size // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)")
    sys.stdout.flush()

urllib.request.urlretrieve(url, output, progress_hook)
print("\nDownload complete!")
"@
    & .\venv\Scripts\python.exe -c $download_code
    Write-Host "Model weights downloaded." -ForegroundColor Green
} else {
    Write-Host "SAM 2.1 Hiera Large weights already exist at $ckpt_file." -ForegroundColor Green
}

Write-Host "`n=========================================" -ForegroundColor Green
Write-Host " SAM 2 Rotoscoper Setup Completed Successfully!" -ForegroundColor Green
Write-Host " Use the virtual environment Python interpreter at .\venv\Scripts\python.exe" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
