@echo off
echo 同步RLSolver仓库...
echo.

REM 检查RLSolver文件夹是否存在
if not exist "RLSolver" (
    echo RLSolver文件夹不存在，正在克隆...
    git clone https://github.com/Open-Finance-Lab/RLSolver.git RLSolver
    if %errorlevel% neq 0 (
        echo 克隆失败，请检查网络连接
        pause
        exit /b 1
    )
) else (
    echo RLSolver文件夹已存在，正在更新...
    cd RLSolver
    git pull origin main
    if %errorlevel% neq 0 (
        echo 更新失败，请检查网络连接
        pause
        exit /b 1
    )
    cd ..
)

echo RLSolver同步完成！
pause