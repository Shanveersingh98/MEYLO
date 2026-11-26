@echo off
title MEYLO — Your AI. Your Space.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment Python not found at: .venv\Scripts\python.exe
    echo Please verify the virtual environment exists before launching.
    pause
    exit /b 1
)

.venv\Scripts\python.exe src\launch.py
exit /b %errorlevel%
