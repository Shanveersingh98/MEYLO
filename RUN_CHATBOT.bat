@echo off
title MEYLO — Your AI. Your Space.
cd /d "%~dp0"

if "%1"=="help" (
    call "%~dp0help.bat"
    exit /b 0
)
if "%1"=="--help" (
    call "%~dp0help.bat"
    exit /b 0
)
if "%1"=="-h" (
    call "%~dp0help.bat"
    exit /b 0
)

echo ===============================================================================
echo Starting MEYLO — Your AI. Your Space.
echo - Independent Dual Models: Custom Trained (best.pt) + Qwen2.5-1.5B-Instruct
echo - Switch models in chat: /model custom  or  /model external
echo - Switch web search    : /web on         or  /web off
echo - Python Environment   : .venv\Scripts\python.exe
echo ===============================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment Python not found at: .venv\Scripts\python.exe
    echo Please ensure the virtual environment exists before launching.
    pause
    exit /b 1
)

.venv\Scripts\python.exe src\chat.py %*
if errorlevel 1 (
    echo.
    echo [NOTICE] Chat application terminated with exit code %errorlevel%.
    pause
)
