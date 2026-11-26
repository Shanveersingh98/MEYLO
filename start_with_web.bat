@echo off
title MEYLO Chatbot (Web Search Enabled)
cd /d "%~dp0"
echo Starting MEYLO with Live Web Search ON...
.venv\Scripts\python.exe src\chat.py --web-search on %*
pause
