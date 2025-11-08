"""
基础环境类
定义了所有环境应该实现的接口
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseEnvironment(ABC):
    """基础环境类，定义了所有环境应该实现的接口"""
    
    def __init__(self, config):
        """
        初始化环境
        
        Args:
            config: 环境配置字典
        """
        self.config = config
        self.current_step = 0
        self.max_steps = config.get('max_steps_per_episode', 40)
        
    @abstractmethod
    def reset(self):
        """
        重置环境到初始状态
        
        Returns:
            observation: 初始观察
        """
        pass
    
    @abstractmethod
    def step(self, action):
        """
        执行一个动作
        
        Args:
            action: 要执行的动作
            
        Returns:
            observation: 下一个观察
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        pass
    
    @abstractmethod
    def get_observation_space(self):
        """
        获取观察空间大小
        
        Returns:
            observation_space: 观察空间的形状
        """
        pass
    
    @abstractmethod
    def get_action_space(self):
        """
        获取动作空间大小
        
        Returns:
            action_space: 动作空间的大小
        """
        pass
    
    @abstractmethod
    def calculate_score(self):
        """
        计算当前状态的得分
        
        Returns:
            score: 当前状态的得分
        """
        pass
    
    def get_best_score(self):
        """
        获取历史最佳得分
        
        Returns:
            best_score: 历史最佳得分
        """
        return getattr(self, 'best_score', 0)
    
    def is_done(self):
        """
        检查回合是否结束
        
        Returns:
            done: 是否结束
        """
        return self.current_step >= self.max_steps


class SpinSystemEnvironment(BaseEnvironment):
    """自旋系统环境，用于最大割等组合优化问题"""
    
    def __init__(self, graph, config):
        """
        初始化自旋系统环境
        
        Args:
            graph: 邻接矩阵表示的图
            config: 环境配置
        """
        super().__init__(config)
        
        self.graph = graph
        self.n_nodes = graph.shape[0]
        self.reversible_spins = config.get('reversible_spins', True)
        self.reward_signal = config.get('reward_signal', 'BLS')
        self.norm_rewards = config.get('norm_rewards', True)
        
        # 初始化状态
        self.spins = None
        self.best_spins = None
        self.best_score = -float('inf')
        self.current_score = 0
        self.init_score = 0
        
        # 动作空间：每个节点可以翻转
        self.action_space = self.n_nodes
        
    def reset(self):
        """重置环境到初始状态"""
        self.current_step = 0
        
        if self.reversible_spins:
            # 可逆自旋：随机初始化为-1或1
            self.spins = np.random.choice([-1, 1], size=self.n_nodes)
        else:
            # 不可逆自旋：初始化为1（表示可以翻转）
            self.spins = np.ones(self.n_nodes)
        
        self.current_score = self._calculate_cut_value()
        self.init_score = self.current_score
        self.best_score = self.current_score
        self.best_spins = self.spins.copy()
        
        return self._get_observation()
    
    def step(self, action):
        """执行一个动作（翻转一个节点）"""
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0, True, {}
        
        self.current_step += 1
        
        # 记录翻转前的分数
        prev_score = self.current_score
        
        # 执行动作：翻转指定节点
        if self.reversible_spins:
            # 可逆模式：直接翻转
            self.spins[action] *= -1
        else:
            # 不可逆模式：只能从1翻转到-1
            if self.spins[action] == 1:
                self.spins[action] = -1
            else:
                # 尝试翻转已经翻转的节点，给予惩罚
                return self._get_observation(), -1, True, {}
        
        # 计算新分数
        self.current_score = self._calculate_cut_value()
        
        # 计算奖励
        reward = self._calculate_reward(prev_score, self.current_score)
        
        # 更新最佳分数
        if self.current_score > self.best_score:
            self.best_score = self.current_score
            self.best_spins = self.spins.copy()
        
        # 检查是否结束
        done = self._check_termination()
        
        return self._get_observation(), reward, done, {}
    
    def get_observation_space(self):
        """获取观察空间大小"""
        # 观察包括：自旋状态 + 邻接矩阵
        return (self.n_nodes, 1 + self.n_nodes)
    
    def get_action_space(self):
        """获取动作空间大小"""
        return self.action_space
    
    def calculate_score(self):
        """计算当前状态的得分"""
        return self._calculate_cut_value()
    
    def _calculate_cut_value(self, spins=None):
        """计算最大割的值"""
        if spins is None:
            spins = self.spins
        
        # 最大割公式：1/4 * sum_{i,j} (1 - s_i * s_j) * w_ij
        cut_value = 0
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                if self.graph[i, j] > 0:
                    cut_value += (1 - spins[i] * spins[j]) * self.graph[i, j]
        
        return cut_value / 4
    
    def _calculate_reward(self, prev_score, current_score):
        """计算奖励"""
        delta_score = current_score - prev_score
        
        if self.reward_signal == 'DENSE':
            # 密集奖励：直接返回分数变化
            reward = delta_score
        elif self.reward_signal == 'BLS':
            # BLS奖励：只有当找到更好解时才给奖励
            reward = delta_score if delta_score > 0 else 0
        else:
            raise ValueError(f"Unknown reward signal: {self.reward_signal}")
        
        # 归一化奖励
        if self.norm_rewards:
            reward = reward / self.n_nodes
        
        return reward
    
    def _get_observation(self):
        """获取当前观察"""
        # 观察包括：自旋状态和邻接矩阵
        obs = np.zeros((self.n_nodes, 1 + self.n_nodes))
        obs[:, 0] = self.spins  # 第一列是自旋状态
        obs[:, 1:] = self.graph  # 剩余列是邻接矩阵
        
        return obs
    
    def _check_termination(self):
        """检查回合是否结束"""
        if self.current_step >= self.max_steps:
            return True
        
        if not self.reversible_spins:
            # 不可逆模式：如果所有节点都翻转了，则结束
            if np.all(self.spins == -1):
                return True
        
        return False