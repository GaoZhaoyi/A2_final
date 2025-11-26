@echo off
echo ==========================================
echo      Auto Push Script for A2 Project
echo ==========================================

echo [1/3] Adding all changes...
git add .

echo [2/3] Committing changes...
git commit -m "Update: Optimize training strategy (50k data, LR 2e-5) and fix Windows crash"

echo [3/3] Pushing to GitHub...
git push

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Push failed! Please check your network or git configuration.
) else (
    echo.
    echo [SUCCESS] Code successfully pushed to GitHub!
)

echo.
pause
