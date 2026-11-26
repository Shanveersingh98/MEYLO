@echo off
title MEYLO Chatbot (Offline Mode)
cd /d "%~dp0"
echo Starting MEYLO in Offline Mode...
.venv\Scripts\python.exe src\chat.py --web-search off %*
pause
