@echo off
title MEYLO Chatbot (Qwen2.5-1.5B Active)
cd /d "%~dp0"
echo Starting MEYLO with External Qwen Model...
.venv\Scripts\python.exe src\chat.py --model external %*
pause
