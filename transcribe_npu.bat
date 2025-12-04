@echo off
setlocal

:: Whisper NPU Transcription Script
:: Uses Snapdragon X Elite NPU via ONNX Runtime QNN

set PYTHON=C:\Users\jimla\AppData\Local\Programs\Python\Python311-arm64\python.exe
set SCRIPT_DIR=%~dp0
set SCRIPT=%SCRIPT_DIR%transcribe_npu.py

:: Add FFmpeg to path if needed
set PATH=%PATH%;C:\ffmpeg\bin;C:\ProgramData\chocolatey\bin

echo.
echo ============================================================
echo  Whisper NPU Transcription - Snapdragon X Elite
echo ============================================================
echo.

:: Run the transcription
%PYTHON% %SCRIPT% %*

endlocal
