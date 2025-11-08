# RLSolver Learning 快速开始指南

本指南将帮助您快速上手 RLSolver 学习版本，在几分钟内运行您的第一个实验。

## 安装

```bash
# 1. 克隆或下载项目
git clone <repository-url> RLSolver_Learning
cd RLSolver_Learning

# 2. 安装依赖
pip install -r requirements.txt
```

## 验证安装

```bash
# 运行基础测试验证安装
python tests/test_basic.py
```

如果看到 "所有测试完成!" 消息，说明安装成功。

## 第一个实验

### 运行ECO算法

```bash
# 运行ECO算法示例
python examples/run_eco.py
```

这将：
1. 生成10个20节点的BA图用于训练
2. 生成5个50节点的BA图用于测试
3. 训练ECO算法100个回合
4. 评估训练后的模型
5. 保存结果到 `models/` 和 `results/` 目录

### 运行S2V算法

```bash
# 运行S2V算法示例
python examples/run_s2v.py
```

这将：
1. 生成10个20节点的BA图用于训练
2. 生成5个50节点的BA图用于测试
3. 训练S2V算法100个回合
4. 评估训练后的模型
5. 保存结果到 `models/` 和 `results/` 目录

## 理解结果

### 训练曲线

训练完成后，您可以在 `results/` 目录中找到：

- `ECO_training_curves.png`：ECO算法的训练曲线
- `S2V_training_curves.png`：S2V算法的训练曲线

### 性能指标

在 `results/` 目录中的JSON文件包含：

- `mean_score`：平均分数
- `best_score`：最佳分数
- `std_score`：分数标准差
- `training_time`：训练时间

## 自定义实验

### 修改配置

编辑 `config.py` 文件来自定义参数：

```python
# 修改训练步数
DEFAULT_CONFIG['num_steps'] = 10000

# 修改图大小
DEFAULT_CONFIG['num_train_nodes'] = 50

# 启用GPU（如果可用）
DEFAULT_CONFIG['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### 使用不同图类型

```python
# 在示例脚本中修改图类型
graphs = [
    generate_random_graph(20, 'ER', seed=i)  # 改为ER图
    for i in range(10)
]
```

### 调整算法参数

```python
# ECO特定配置
eco_config = create_eco_config(DEFAULT_CONFIG)
eco_config.update({
    'gamma': 0.99,  # 更高的折扣因子
    'learning_rate': 5e-4,  # 更低的学习率
})

# S2V特定配置
s2v_config = create_s2v_config(DEFAULT_CONFIG)
s2v_config.update({
    'epsilon_decay_steps': 10000,  # 更慢的探索衰减
})
```

## 常见问题

### Q: 训练很慢怎么办？

A: 尝试以下优化：
1. 减少训练图的大小
2. 减少训练回合数
3. 使用GPU而不是CPU
4. 减少网络复杂度

### Q: 内存不足怎么办？

A: 尝试以下优化：
1. 减少经验回放缓冲区大小
2. 减少批量大小
3. 减少环境并行数
4. 使用梯度累积

### Q: 如何可视化结果？

A: 训练完成后：
1. 查看 `results/` 目录中的PNG文件
2. 使用 `visualize_graph()` 函数可视化图和解
3. 使用matplotlib或其他工具绘制自定义图表

## 进阶使用

### 添加自定义算法

1. 在 `src/algorithms/` 中创建新算法文件
2. 继承 `BaseAlgorithm` 类
3. 实现必要的方法
4. 在示例脚本中测试

### 添加新问题

1. 在 `src/environments/` 中创建新环境文件
2. 继承 `BaseEnvironment` 类
3. 实现问题特定的逻辑
4. 修改现有算法以支持新问题

### 性能基准测试

```bash
# 运行性能比较
python examples/run_s2v.py  # 包含比较函数
```

## 下一步

1. 阅读 `docs/tutorial.md` 了解详细概念
2. 查看 `examples/` 中的更多示例
3. 修改 `config.py` 自定义实验
4. 在 `src/` 中添加自己的算法

## 支持

如果遇到问题：

1. 查看 `docs/tutorial.md` 中的常见问题
2. 检查 `tests/` 中的测试用例
3. 查看代码注释了解实现细节

祝您使用愉快！