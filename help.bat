@echo off
title MEYLO - Command Guide and Launcher
cd /d "%~dp0"

echo ===============================================================================
echo                             MEYLO - COMMAND GUIDE
echo                             Your AI. Your Space.
echo                     Project: D:\DEEP-LEARNING-PROJECT
echo ===============================================================================
echo.
echo [1] QUICK LAUNCH COMMANDS:
echo   - Default (Custom Model, Offline):
echo       .venv\Scripts\python.exe src\chat.py
echo.
echo   - External Qwen Model (Offline):
echo       .venv\Scripts\python.exe src\chat.py --model external
echo.
echo   - Live Web Search ON (Custom Model):
echo       .venv\Scripts\python.exe src\chat.py --web-search on
echo.
echo   - Live Web Search ON + External Qwen Model:
echo       .venv\Scripts\python.exe src\chat.py --model external --web-search on
echo.
echo [2] ONE-CLICK BATCH SCRIPTS:
echo   - START_AI_COMPANION.bat       : Launches Modern Web UI in default browser
echo   - START_AI_COMPANION_DEBUG.bat : Launches Web UI with live diagnostic console
echo   - STOP_AI_COMPANION.bat        : Safely terminates all AI Companion services
echo   - RUN_CHATBOT.bat              : Starts terminal chatbot (Custom Model, Offline)
echo   - start_qwen.bat               : Starts terminal with External Qwen Model active
echo   - start_with_web.bat           : Starts terminal with Live Web Search ON
echo   - start_without_web.bat        : Starts terminal in pure Offline Mode
echo.
echo [3] IN-CHAT COMMANDS (Once Chatbot is Running):
echo   - /model custom   : Switch to Custom Trained Model (best.pt on CUDA GPU)
echo   - /model external : Switch to External Qwen Model (on multi-threaded CPU)
echo   - /model status   : Check active model, RAM, and GPU VRAM usage
echo   - /model list     : List available models and active status
echo   - /web on         : Turn live web search ON
echo   - /web off        : Turn live web search OFF (offline companion mode)
echo   - /web status     : Check whether web search is ON or OFF
echo   - /sources        : View clickable URLs of retrieved web sources
echo   - /clear          : Reset conversation memory and active subject
echo   - /help           : Show in-chat help menu
echo   - /exit           : Cleanly release GPU/RAM memory and close chatbot
echo.
echo ===============================================================================
echo.
echo Select an option to launch now, or press [0] to exit back to command prompt:
echo   [1] Launch Terminal Chatbot (Custom Model, Offline)
echo   [2] Launch Terminal with External Qwen Model
echo   [3] Launch Terminal with Live Web Search ON (Custom)
echo   [4] Launch Terminal with Qwen + Live Web Search ON
echo   [5] Launch Modern Web UI Application (START_AI_COMPANION.bat)
echo   [0] Exit back to prompt
echo.
set /p opt="Enter your choice [0-5]: "

if "%opt%"=="1" (
    echo.
    .venv\Scripts\python.exe src\chat.py
) else if "%opt%"=="2" (
    echo.
    .venv\Scripts\python.exe src\chat.py --model external
) else if "%opt%"=="3" (
    echo.
    .venv\Scripts\python.exe src\chat.py --web-search on
) else if "%opt%"=="4" (
    echo.
    .venv\Scripts\python.exe src\chat.py --model external --web-search on
) else if "%opt%"=="5" (
    echo.
    call "%~dp0START_AI_COMPANION.bat"
) else (
    echo.
)
