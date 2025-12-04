@echo off
setlocal

:: Whisper Hybrid Transcription Script
:: NPU Encoder + CPU Decoder (with KV-cache) for Snapdragon X Elite

set PYTHON=C:\Users\jimla\AppData\Local\Programs\Python\Python311-arm64\python.exe
set SCRIPT_DIR=%~dp0
set SCRIPT=%SCRIPT_DIR%transcribe_hybrid.py

echo.
echo ============================================================
echo  Whisper Hybrid Transcription - Snapdragon X Elite
echo  NPU Encoder + CPU Decoder (with KV-cache)
echo ============================================================
echo.

:: Run the transcription
%PYTHON% %SCRIPT% %*

endlocal
