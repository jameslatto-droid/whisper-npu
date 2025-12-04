#Requires -Version 5.1
<#
.SYNOPSIS
    Setup script for Whisper NPU transcription on Snapdragon X Elite
.DESCRIPTION
    This script installs all dependencies and downloads the Whisper ONNX model
    for NPU-accelerated transcription using Qualcomm QNN SDK.
#>

param(
    [string]$ModelSize = "small",  # tiny, base, small, medium
    [switch]$SkipPythonCheck,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
$ModelsDir = Join-Path $ProjectDir "models"
$ScriptsDir = Join-Path $ProjectDir "scripts"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Whisper NPU Setup for Snapdragon X Elite  " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check architecture
$arch = $env:PROCESSOR_ARCHITECTURE
if ($arch -ne "ARM64") {
    Write-Warning "This script is designed for ARM64. Detected: $arch"
}
Write-Host "[OK] Architecture: $arch" -ForegroundColor Green

# Create directories
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
New-Item -ItemType Directory -Force -Path $ScriptsDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "temp") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "transcripts") | Out-Null

Write-Host "[OK] Created project directories" -ForegroundColor Green

# Check Python
if (-not $SkipPythonCheck) {
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python 3\.(\d+)") {
            $minor = [int]$Matches[1]
            if ($minor -ge 9) {
                Write-Host "[OK] Python: $pythonVersion" -ForegroundColor Green
            } else {
                throw "Python 3.9+ required"
            }
        }
    } catch {
        Write-Host "[!!] Python not found or too old" -ForegroundColor Red
        Write-Host ""
        Write-Host "Install Python 3.11 ARM64:" -ForegroundColor Yellow
        Write-Host "  winget install Python.Python.3.11 --architecture arm64" -ForegroundColor White
        Write-Host ""
        exit 1
    }
}

# Check for QNN SDK
$qnnRoot = $env:QNN_SDK_ROOT
if (-not $qnnRoot -or -not (Test-Path $qnnRoot)) {
    Write-Host "[!!] QNN SDK not found" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To use NPU acceleration, install Qualcomm AI Engine Direct:" -ForegroundColor Yellow
    Write-Host "  1. Visit: https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Engine_Direct" -ForegroundColor White
    Write-Host "  2. Download the Windows ARM64 SDK" -ForegroundColor White
    Write-Host "  3. Extract to C:\Qualcomm\AIStack\QAIRT\" -ForegroundColor White
    Write-Host "  4. Set QNN_SDK_ROOT environment variable" -ForegroundColor White
    Write-Host ""
    Write-Host "Continuing with ONNX Runtime (CPU/DirectML) fallback..." -ForegroundColor Yellow
    $useQNN = $false
} else {
    Write-Host "[OK] QNN SDK: $qnnRoot" -ForegroundColor Green
    $useQNN = $true
}

# Check for ffmpeg
try {
    $ffmpegVersion = ffmpeg -version 2>&1 | Select-Object -First 1
    Write-Host "[OK] FFmpeg: Found" -ForegroundColor Green
} catch {
    Write-Host "[!!] FFmpeg not found" -ForegroundColor Yellow
    Write-Host "Installing FFmpeg via winget..." -ForegroundColor Yellow
    winget install Gyan.FFmpeg
}

# Install Python packages
Write-Host ""
Write-Host "Installing Python packages..." -ForegroundColor Cyan

$packages = @(
    "numpy",
    "onnxruntime-directml",  # DirectML for GPU fallback
    "transformers",
    "librosa",
    "soundfile",
    "tqdm"
)

foreach ($pkg in $packages) {
    Write-Host "  Installing $pkg..." -ForegroundColor Gray
    pip install $pkg --quiet --disable-pip-version-check 2>$null
}
Write-Host "[OK] Python packages installed" -ForegroundColor Green

# Download Whisper ONNX model
Write-Host ""
Write-Host "Downloading Whisper ONNX model ($ModelSize)..." -ForegroundColor Cyan

$modelUrls = @{
    "tiny"   = "https://huggingface.co/onnx-community/whisper-tiny/resolve/main"
    "base"   = "https://huggingface.co/onnx-community/whisper-base/resolve/main"
    "small"  = "https://huggingface.co/onnx-community/whisper-small/resolve/main"
    "medium" = "https://huggingface.co/onnx-community/whisper-medium/resolve/main"
}

$modelFiles = @(
    "encoder_model.onnx",
    "decoder_model_merged.onnx",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "preprocessor_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "normalizer.json",
    "merges.txt"
)

$modelDir = Join-Path $ModelsDir "whisper-$ModelSize"
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

$baseUrl = $modelUrls[$ModelSize]
foreach ($file in $modelFiles) {
    $outPath = Join-Path $modelDir $file
    if ((Test-Path $outPath) -and -not $Force) {
        Write-Host "  [skip] $file (exists)" -ForegroundColor Gray
        continue
    }
    
    $url = "$baseUrl/$file"
    Write-Host "  Downloading $file..." -ForegroundColor Gray
    try {
        Invoke-WebRequest -Uri $url -OutFile $outPath -UseBasicParsing
    } catch {
        Write-Host "  [warn] Could not download $file" -ForegroundColor Yellow
    }
}

Write-Host "[OK] Model downloaded to: $modelDir" -ForegroundColor Green

# Create summary
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!                          " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project directory: $ProjectDir" -ForegroundColor White
Write-Host "Model directory:   $modelDir" -ForegroundColor White
Write-Host "NPU acceleration:  $(if ($useQNN) { 'Available' } else { 'Not available (using DirectML)' })" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run: .\transcribe_npu.ps1 'C:\path\to\audio.mp3'" -ForegroundColor White
Write-Host "  2. Or batch: .\transcribe_npu.ps1 'C:\path\to\folder'" -ForegroundColor White
Write-Host ""
