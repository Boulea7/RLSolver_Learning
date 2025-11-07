@echo off
echo ========================================
echo 完整同步RLSolver_Learning项目
echo ========================================
echo.

REM 检查Git状态
echo 检查当前Git状态...
git status
echo.

REM 添加所有更改
echo 添加所有更改...
git add .
echo.

REM 提交更改
echo 提交更改...
git commit -m "项目同步：更新.gitignore、添加同步脚本和说明文档"
echo.

REM 尝试推送到远程仓库
echo 尝试推送到GitHub...
git push origin main
if %errorlevel% neq 0 (
    echo.
    echo 推送失败！可能是网络连接问题。
    echo 请在网络恢复后手动运行：git push origin main
    echo.
)

REM 删除远程master分支（如果存在）
echo.
echo 尝试删除远程master分支...
git push origin --delete master
if %errorlevel% neq 0 (
    echo 删除master分支失败（可能已不存在或网络问题）
)

REM 同步RLSolver子仓库
echo.
echo 同步RLSolver参考仓库...
if not exist "RLSolver" (
    echo RLSolver文件夹不存在，正在克隆...
    git clone https://github.com/Open-Finance-Lab/RLSolver.git RLSolver
    if %errorlevel% neq 0 (
        echo 克隆RLSolver失败，请检查网络连接
    )
) else (
    echo RLSolver文件夹已存在，正在更新...
    cd RLSolver
    git pull origin main
    if %errorlevel% neq 0 (
        echo 更新RLSolver失败，请检查网络连接
    )
    cd ..
)

echo.
echo ========================================
echo 同步操作完成！
echo ========================================
echo.
echo 注意事项：
echo 1. 如果推送失败，请在网络恢复后手动运行 git push origin main
echo 2. RLSolver已设置为独立仓库，不会被主仓库跟踪
echo 3. 可以定期运行 sync_rlsolver.bat 来更新RLSolver
echo.
pause