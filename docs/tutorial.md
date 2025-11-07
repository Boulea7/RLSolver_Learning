
# RLSolver 学习教程

本教程将引导您了解和使用 RLSolver 学习版本，深入理解 ECO 和 S2V 算法的核心逻辑。

## 目录

1. [环境设置](#环境设置)
2. [基础概念](#基础概念)
3. [ECO 算法](#eco-算法)
4. [S2V 算法](#s2v-算法)
5. [比较分析](#比较分析)
6. [扩展开发](#扩展开发)

## 环境设置

### 安装依赖

```bash
# 进入项目目录
cd RLSolver_Learning

# 安装依赖
pip install -r requirements.txt
```

### 验证安装

```bash
# 运行基础测试
python tests/test_basic.py
```

如果所有测试都通过，说明环境设置正确。

## 基础概念

### 最大割问题

最大割问题是图论中的一个经典组合优化问题：

- **输入**：无向图 G = (V, E)，其中 V 是顶点集，E 是边集
- **目标**：将顶点划分为两个集合 A 和 B，最大化 A 和 B 之间的边权重和
- **数学表达**：最大化 Σ_{(u,v)∈E, u∈A, v∈B} w(u,v)

### 强化学习框架

我们将最大割问题构建为强化学习问题：

- **状态**：当前顶点划分
- **动作**：选择一个顶点并改变其归属
- **奖励**：割值的变化或当前割值
- **策略**：从状态到动作的映射

### 两种模式

#### 1. 稀疏奖励模式 (ECO)

- **特点**：只有在找到更好解时才给予奖励
- **环境**：可逆自旋系统，可以多次翻转同一顶点
- **优势**：探索更稳定，适合长期优化
- **适用场景**：需要高质量解的应用

#### 2. 密集奖励模式 (S2V)

- **特点**：每次动作都给予即时奖励
- **环境**：不可逆自旋系统，每个顶点只能翻转一次
- **优势**：学习更快，收敛更迅速
- **适用场景**：需要快速求解的应用

## ECO 算法

### 核心思想

ECO (Efficient Combinatorial Optimization) 使用稀疏奖励和可逆自旋系统：

1. **状态表示**：自旋状态 + 邻接矩阵
2. **动作选择**：使用 DQN 选择要翻转的顶点
3. **奖励机制**：BLS (Baseline Learning with Sampling)
4. **探索策略**：ε-贪婪，逐渐减少探索

### 运行示例

```bash
# 运行ECO算法示例
python examples/run_eco.py
```

### 关键参数

- `reversible_spins`: True (可逆自旋)
- `reward_signal`: 'BLS' (稀疏奖励)
- `gamma`: 0.95 (折扣因子)
- `max_steps_per_episode`: 2 * 节点数

### 学习曲线分析

ECO 算法的学习曲线通常显示：

- 初期：分数波动较大，探索阶段
- 中期：分数逐渐提升，学习阶段
- 后期：分数趋于稳定，收敛阶段

## S2V 算法

### 核心思想

S2V (Structure2Vector) 使用密集奖励和不可逆自旋系统：

1. **状态表示**：自旋状态 + 邻接矩阵
2. **动作选择**：使用 DQN 选择要翻转的顶点
3. **奖励机制**：即时奖励等于割值变化
4. **探索策略**：ε-贪婪，逐渐减少探索

### 运行示例

```bash
# 运行S2V算法示例
python examples/run_s2v.py
```

### 关键参数

- `reversible_spins`: False (不可逆自旋)
- `reward_signal`: 'DENSE' (密集奖励)
- `gamma`: 1.0 (无折扣)
- `max_steps_per_episode`: 节点数

### 学习曲线分析

S2V 算法的学习曲线通常显示：

- 初期：分数快速提升，密集奖励引导学习
- 中期：分数稳定提升，逐步优化
- 后期：分数收敛，找到局部最优

## 比较分析

### 算法对比

| 特性 | ECO | S2V |
|-------|------|------|
| 奖励类型 | 稀疏 | 密集 |
| 自旋系统 | 可逆 | 不可逆 |
| 探索能力 | 强 | 中等 |
| 收敛速度 | 慢 | 快 |
| 解质量 | 高 | 中等 |
| 计算效率 | 中等 | 高 |

### 适用场景

- **ECO**：适合需要高质量解的场景，如最终求解
- **S2V**：适合需要快速求解的场景，如初步分析

### 运行比较

```bash
# 比较ECO和S2V算法
python examples/run_s2v.py  # 包含比较函数
```

## 扩展开发

### 添加新算法

1. **继承基类**：
   ```python
   from src.algorithms.base import BaseAlgorithm
   
   class NewAlgorithm(BaseAlgorithm):
       def __init__(self, config):
           super().__init__(config)
           # 初始化算法特定参数
   ```

2. **实现核心方法**：
   ```python
   def create_environment(self, graph):
       # 创建算法特定环境
       pass
   
   def create_agent(self, observation_space, action_space):
       # 创建算法特定智能体
       pass
   
   def train(self, graphs, num_episodes):
       # 实现训练逻辑
       pass
   ```

3. **添加配置**：
   ```python
   # 在config.py中添加新算法类型
   class AlgorithmType(Enum):
       NEW_ALGO = 'NEW_ALGO'
   ```

### 添加新问题

1. **扩展环境**：
   ```python
   from src.environments.base import BaseEnvironment
   
   class NewProblemEnvironment(BaseEnvironment):
       def __init__(self, problem_instance, config):
           super().__init__(config)
           # 初始化问题特定参数
   ```

2. **实现问题方法**：
   ```python
   def calculate_score(self):
       # 实现问题特定的评分函数
       pass
   
   def get_solution(self):
       # 返回问题特定的解格式
       pass
   ```

### 性能优化建议

1. **批量处理**：使用向量化操作提高效率
2. **内存管理**：合理设置经验回放缓冲区大小
3. **并行化**：利用多GPU或多进程加速训练
4. **缓存计算**：缓存重复计算的结果

## 常见问题

### Q: 训练不收敛怎么办？

A: 检查以下几点：
- 学习率是否合适（尝试降低）
- 网络容量是否足够（增加层数或特征数）
- 奖励设计是否合理（检查奖励分布）
- 探索策略是否有效（调整ε衰减）

### Q: 如何选择合适的图大小？

A: 根据计算资源和时间限制：
- 小图（<50节点）：快速原型验证
- 中等图（50-200节点）：平衡性能和质量
- 大图（>200节点）：需要更多计算资源

### Q: 如何改进解质量？

A: 尝试以下方法：
- 增加训练回合数
- 使用更大的网络
- 调整奖励函数
- 集成局部搜索
- 使用集成方法

## 进阶主题

### 注意力机制

在图神经网络中引入注意力机制可以提高性能：

```python
class AttentionQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(...)
        self.network = nn.Sequential(...)
    
    def forward(self, x):
        # 应用注意力机制
        attended = self.attention(x, x, x)
        return self.network(attended)
```

### 元学习

使用元学习快速适应新的图：

```python
class MetaLearner:
    def __init__(self, base_model):
        self.base_model = base_model
        self.meta_optimizer = ...
    
    def adapt(self, support_tasks):
        # 在支持任务上快速适应
        for task in support_tasks:
            self.fast_adapt(task)
```

### 分布式训练

对于大规模训练，使用分布式训练：

```python
import torch.distributed as dist

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    
def distributed_train(model, dataloader):
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank]
    )
    # 分布式训练逻辑
```

## 总结

本教程涵盖了 RLSolver 学习版本的核心概念和使用方法。通过理解 ECO 和 S2V 算法的差异，您可以选择适合特定应用场景的算法，并基于现有