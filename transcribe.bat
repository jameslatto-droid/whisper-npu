@echo off
setlocal

REM Set Python path explicitly
set PYTHON=C:\Users\jimla\AppData\Local\Programs\Python\Python311-arm64\python.exe

REM Add FFmpeg to PATH
set PATH=%PATH%;C:\Users\jimla\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin

REM Check if Python exists
if not exist "%PYTHON%" (
    echo Python not found at %PYTHON%
    echo Please update the PYTHON path in this script.
    pause
    exit /b 1
)

REM Run the transcription script with all arguments passed through
"%PYTHON%" "%~dp0transcribe_ffmpeg.py" %*
