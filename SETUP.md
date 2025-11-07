# 项目设置说明

## 概述

这个项目是RLSolver_Learning，主仓库位于GitHub上。项目中包含一个参考用的RLSolver仓库，该仓库来自Open-Finance-Lab/RLSolver。

## 仓库结构

- 主仓库：`https://github.com/Boulea7/RLSolver_Learning.git` (我们自己的仓库)
- 参考仓库：`https://github.com/Open-Finance-Lab/RLSolver.git` (只读参考)

## Git分支策略

- 只使用`main`分支，其他分支将被删除
- 本地代码优先，以本地为准进行同步

## RLSolver管理

由于RLSolver是外部参考仓库，我们需要保持其与上游同步，但不将其包含在主仓库的Git历史中。

### 方法1：使用同步脚本（推荐）

运行提供的同步脚本来更新RLSolver：
```bash
# 在Windows上
sync_rlsolver.bat

# 或者在Linux/Mac上
./sync_rlsolver.sh
```

### 方法2：手动同步

```bash
# 如果RLSolver文件夹不存在
git clone https://github.com/Open-Finance-Lab/RLSolver.git RLSolver

# 如果已存在，进入文件夹更新
cd RLSolver
git pull origin main
cd ..
```

### 方法3：使用Git子模块（网络恢复后）

```bash
# 删除现有的RLSolver文件夹（如果有）
rmdir /s /q RLSolver  # Windows
rm -rf RLSolver        # Linux/Mac

# 添加为子模块
git submodule add https://github.com/Open-Finance-Lab/RLSolver.git RLSolver

# 初始化子模块
git submodule init

# 更新子模块
git submodule update
```

## 初始设置步骤

1. 克隆主仓库：
   ```bash
   git clone https://github.com/Boulea7/RLSolver_Learning.git
   cd RLSolver_Learning
   ```

2. 同步RLSolver：
   ```bash
   sync_rlsolver.bat  # Windows
   ```

3. 删除其他分支（如果存在）：
   ```bash
   git branch -d master  # 删除本地master分支
   git push origin --delete master  # 删除远程master分支
   ```

## 常规维护

### 定期同步RLSolver

定期运行同步脚本以获取最新的RLSolver更新：
```bash
sync_rlsolver.bat
```

### 推送主仓库更改

```bash
git add .
git commit -m "你的提交信息"
git push origin main
```

## 注意事项

- RLSolver文件夹已在.gitignore中被忽略，不会被主仓库跟踪
- 确保不要将RLSolver中的代码直接复制到主仓库中
- 如果需要使用RLSolver的功能，请通过导入或引用的方式使用
- 网络连接问题可能会影响同步操作，请在网络稳定时执行同步