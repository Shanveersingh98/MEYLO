@echo off
title Stopping MEYLO Services...
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe src\stop.py
) else (
    taskkill /IM python.exe /FI "WINDOWTITLE eq AI Companion*" /F >nul 2>&1
    taskkill /IM llama-server.exe /F >nul 2>&1
    echo [Shutdown] Stopped processes via Windows fallback.
)

ping -n 3 127.0.0.1 >nul
exit /b 0
