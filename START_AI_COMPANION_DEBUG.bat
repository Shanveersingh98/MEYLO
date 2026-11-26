@echo off
title MEYLO Diagnostics & Live Server Console
cd /d "%~dp0"

echo ===============================================================================
echo                   MEYLO (DIAGNOSTIC & LOGS CONSOLE)
echo                         Your AI. Your Space.
echo ===============================================================================
echo - Dual Models: Custom Trained (best.pt) + Qwen2.5-1.5B (GGUF)
echo - Server URL : http://127.0.0.1:8000
echo - To Stop    : Press Ctrl+C in this window or run STOP_AI_COMPANION.bat
echo ===============================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment Python not found at: .venv\Scripts\python.exe
    pause
    exit /b 1
)

.venv\Scripts\python.exe src\api_server.py --open-browser
if errorlevel 1 (
    echo.
    echo [NOTICE] AI Companion API server stopped with exit code %errorlevel%.
    pause
)
