<# 
.SYNOPSIS
    Batch transcribe audio files using Whisper AI
    
.DESCRIPTION
    This script transcribes audio files (.mp3, .m4a, .wav, etc.) using OpenAI Whisper.
    Optimized for Windows ARM64 / Snapdragon X Elite.

.EXAMPLE
    .\transcribe.ps1 -InputPath "C:\Audio\recording.mp3"
    .\transcribe.ps1 -InputPath "C:\Audio\" -Model medium -Language en
    .\transcribe.ps1 -InputPath "C:\Audio\" -OutputPath "C:\Transcripts\"
#>

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$InputPath,
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("tiny", "base", "small", "medium", "large")]
    [string]$Model = "base",
    
    [Parameter(Mandatory=$false)]
    [string]$Language = $null,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputPath = $null,
    
    [Parameter(Mandatory=$false)]
    [string]$Format = "txt,srt"
)

$ErrorActionPreference = "Stop"

# Banner
Write-Host "=" * 50 -ForegroundColor Cyan
Write-Host "🎙️  Whisper Transcription Tool" -ForegroundColor Cyan
Write-Host "   Windows ARM64 / Snapdragon X Elite" -ForegroundColor Gray
Write-Host "=" * 50 -ForegroundColor Cyan

# Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "❌ Python not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

# Check if whisper is installed
$whisperInstalled = python -c "import whisper" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "⏳ Installing Whisper... (this may take a few minutes)" -ForegroundColor Yellow
    pip install openai-whisper
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install Whisper" -ForegroundColor Red
        exit 1
    }
}

# Build arguments
$scriptPath = Join-Path $PSScriptRoot "transcribe.py"
$args = @($scriptPath, "`"$InputPath`"", "--model", $Model, "--format", $Format)

if ($Language) {
    $args += @("--language", $Language)
}

if ($OutputPath) {
    $args += @("--output", "`"$OutputPath`"")
}

# Check if it's a folder
if (Test-Path $InputPath -PathType Container) {
    $args += "--batch"
}

# Run transcription
Write-Host ""
Write-Host "Starting transcription..." -ForegroundColor Green
Write-Host "Input: $InputPath" -ForegroundColor Gray
Write-Host "Model: $Model" -ForegroundColor Gray
Write-Host ""

& python @args

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Transcription complete!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "❌ Transcription failed" -ForegroundColor Red
    exit 1
}
