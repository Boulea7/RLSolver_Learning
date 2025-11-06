# RLSolver 学习与简化版本

## 项目概述

本项目是基于 [RLSolver](https://github.com/Open-Finance-Lab/RLSolver) 的学习和简化版本，旨在深入理解 ECO 和 S2V 算法的核心逻辑，并提供一个最小化但功能完整的强化学习求解组合优化问题的框架。

## 项目目标

1. **简化理解**：深入理解 ECO 和 S2V 算法的核心逻辑，减少冗余代码
2. **最小化实现**：创建一个仅包含核心功能的 MVP，确保代码能最简化地运行
3. **减少依赖**：分析每个库的作用，减少项目所依赖的第三方库
4. **可扩展性**：为未来集成新算法或问题预留空间

## 核心算法分析

### ECO (Efficient Combinatorial Optimization)
- **模式**：稀疏奖励模式 (Pattern I)
- **核心思想**：将组合优化问题构建为马尔可夫决策过程 (MDP)
- **奖励机制**：使用 BLS (Baseline Learning with Sampling) 奖励信号
- **环境特性**：可逆自旋系统，允许多次翻转同一个节点

### S2V (Structure2Vector)
- **模式**：密集奖励模式 (Pattern II)
- **核心思想**：学习一个策略来最小化哈密顿目标函数
- **奖励机制**：密集奖励，每次操作立即获得奖励
- **环境特性**：不可逆自旋系统，每个节点只能翻转一次

## 项目结构

```
RLSolver_Learning/
├── src/                    # 核心源代码
│   ├── algorithms/          # 算法实现
│   │   ├── eco.py          # ECO算法简化版
│   │   └── s2v.py         # S2V算法简化版
│   ├── environments/        # 环境实现
│   │   ├── maxcut.py       # 最大割问题环境
│   │   └── base.py        # 基础环境类
│   ├── agents/             # 智能体实现
│   │   ├── dqn.py         # 简化DQN智能体
│   │   └── base.py        # 基础智能体类
│   └── utils/             # 工具函数
│       ├── graph.py        # 图处理工具
│       └── train.py       # 训练工具
├── data/                  # 数据文件
├── examples/              # 示例脚本
├── tests/                 # 测试脚本
├── docs/                  # 文档
├── requirements.txt        # 依赖列表
└── config.py             # 配置文件
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
# 运行ECO算法示例
python examples/run_eco.py

# 运行S2V算法示例
python examples/run_s2v.py
```

## 核心组件

### 1. 环境 (Environment)
- **MaxCutEnvironment**：最大割问题的环境实现
- 支持可逆和不可逆两种模式
- 提供稀疏和密集两种奖励信号

### 2. 智能体 (Agent)
- **DQNAgent**：简化的深度Q网络智能体
- 支持经验回放和目标网络
- 实现ε-贪婪探索策略

### 3. 算法 (Algorithm)
- **ECOAlgorithm**：ECO算法的简化实现
- **S2VAlgorithm**：S2V算法的简化实现

## 学习路线

1. **基础理解**：从 `examples/` 中的简单示例开始
2. **算法分析**：阅读 `src/algorithms/` 中的简化算法实现
3. **环境交互**：了解 `src/environments/` 中的环境设计
4. **智能体训练**：学习 `src/agents/` 中的智能体训练过程
5. **扩展开发**：基于现有框架开发新的算法或问题

## 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目基于原 RLSolver 项目的许可证，详见 LICENSE 文件。

## 致谢

感谢 [Open-Finance-Lab/RLSolver](https://github.com/Open-Finance-Lab/RLSolver) 项目提供的原始实现和理论基础。