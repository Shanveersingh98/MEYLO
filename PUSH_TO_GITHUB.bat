@echo off
title Push MEYLO to GitHub
cd /d "%~dp0"
echo ===============================================================================
echo                           PUSH MEYLO TO GITHUB
echo ===============================================================================
echo Repository: https://github.com/Shanveersingh98/MEYLO.git
echo Branch:     main
echo.
echo Pushing commits to GitHub...
echo (If Git Credential Manager prompts you, click "Sign in with your browser")
echo.
git push -u origin main
echo.
if %ERRORLEVEL% equ 0 (
    echo ===============================================================================
    echo [SUCCESS] MEYLO was successfully uploaded to GitHub!
    echo View repository: https://github.com/Shanveersingh98/MEYLO
    echo ===============================================================================
) else (
    echo ===============================================================================
    echo [INFO] If authentication failed, you can also push with a GitHub Token:
    echo git push https://YOUR_TOKEN@github.com/Shanveersingh98/MEYLO.git main
    echo ===============================================================================
)
echo.
pause
