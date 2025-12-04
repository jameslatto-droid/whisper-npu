Write-Host "====================================" -ForegroundColor Cyan
Write-Host "Installing Whisper dependencies" -ForegroundColor Cyan
Write-Host "This takes 10-15 minutes for Rust builds" -ForegroundColor Yellow
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Continue"

# Install packages one by one with progress
$packages = @(
    "safetensors",
    "tokenizers", 
    "transformers"
)

foreach ($pkg in $packages) {
    Write-Host "Installing $pkg..." -ForegroundColor Green
    $startTime = Get-Date
    pip install $pkg 2>&1 | Out-Null
    $elapsed = (Get-Date) - $startTime
    Write-Host "  Done in $($elapsed.TotalSeconds.ToString('0.0'))s" -ForegroundColor Gray
}

Write-Host ""
Write-Host "====================================" -ForegroundColor Green
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green

# Verify
Write-Host ""
Write-Host "Installed packages:" -ForegroundColor Cyan
pip list | Select-String "safetensor|tokenizer|transform"
