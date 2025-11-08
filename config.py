"""
RLSolver Learning 配置文件
包含算法、环境和训练的默认参数
"""

from enum import Enum
import os

# 算法类型
class AlgorithmType(Enum):
    ECO = 'ECO'
    S2V = 'S2V'

# 问题类型
class ProblemType(Enum):
    MAXCUT = 'maxcut'

# 图类型
class GraphType(Enum):
    BA = 'BA'  # Barabási-Albert 图
    ER = 'ER'  # Erdős-Rényi 图

# 奖励信号类型
class RewardSignal(Enum):
    BLS = 'BLS'  # Baseline Learning with Sampling (稀疏奖励)
    DENSE = 'DENSE'  # 密集奖励

# 默认配置
DEFAULT_CONFIG = {
    # 算法配置
    'algorithm': AlgorithmType.ECO,
    'problem': ProblemType.MAXCUT,
    'graph_type': GraphType.BA,
    
    # 训练参数
    'num_train_nodes': 20,  # 训练图的节点数
    'num_train_envs': 32,  # 训练环境数量
    'num_validation_nodes': 20,  # 验证图的节点数
    'num_validation_envs': 16,  # 验证环境数量
    'num_steps': 10000,  # 训练步数
    'max_steps_per_episode': 40,  # 每个回合最大步数
    
    # DQN参数
    'gamma': 0.95,  # 折扣因子
    'learning_rate': 1e-3,  # 学习率
    'replay_buffer_size': 10000,  # 经验回放缓冲区大小
    'replay_start_size': 1000,  # 开始训练的最小经验数
    'minibatch_size': 32,  # 小批量大小
    'update_frequency': 32,  # 更新频率
    'update_target_frequency': 1000,  # 目标网络更新频率
    'epsilon_start': 1.0,  # 初始探索率
    'epsilon_end': 0.05,  # 最终探索率
    'epsilon_decay_steps': 5000,  # 探索率衰减步数
    
    # 环境参数
    'reversible_spins': True,  # 是否可逆自旋 (ECO=True, S2V=False)
    'reward_signal': RewardSignal.BLS,  # 奖励信号类型
    'norm_rewards': True,  # 是否归一化奖励
    
    # 网络参数
    'network_layers': 3,  # 网络层数
    'network_features': 64,  # 每层特征数
    'init_weight_std': 0.01,  # 初始权重标准差
    
    # 保存和日志
    'save_frequency': 1000,  # 保存频率
    'log_frequency': 100,  # 日志频率
    'test_frequency': 500,  # 测试频率
    'device': 'cpu',  # 计算设备 ('cpu' 或 'cuda')
    
    # 路径配置
    'data_dir': 'data',
    'model_dir': 'models',
    'log_dir': 'logs',
    'result_dir': 'results',
}

# 获取配置值的辅助函数
def get_config(key, default=None):
    """获取配置值，如果不存在则返回默认值"""
    return DEFAULT_CONFIG.get(key, default)

# 设置配置值的辅助函数
def set_config(key, value):
    """设置配置值"""
    DEFAULT_CONFIG[key] = value

# 从环境变量加载配置
def load_from_env():
    """从环境变量加载配置"""
    env_mappings = {
        'RLS_DEVICE': 'device',
        'RLS_ALGORITHM': 'algorithm',
        'RLS_GRAPH_TYPE': 'graph_type',
        'RLS_NUM_STEPS': 'num_steps',
    }
    
    for env_var, config_key in env_mappings.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            # 尝试转换为适当的类型
            if config_key in ['device', 'algorithm', 'graph_type']:
                set_config(config_key, value)
            elif config_key in ['num_steps']:
                set_config(config_key, int(value))