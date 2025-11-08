"""
基础智能体类
定义了所有智能体应该实现的接口
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """基础智能体类，定义了所有智能体应该实现的接口"""
    
    def __init__(self, config):
        """
        初始化智能体
        
        Args:
            config: 智能体配置字典
        """
        self.config = config
        self.training = True
        
    @abstractmethod
    def act(self, observation):
        """
        根据观察选择动作
        
        Args:
            observation: 当前观察
            
        Returns:
            action: 选择的动作
        """
        pass
    
    @abstractmethod
    def learn(self, experience):
        """
        从经验中学习
        
        Args:
            experience: 经验元组 (state, action, reward, next_state, done)
        """
        pass
    
    @abstractmethod
    def save(self, path):
        """
        保存智能体模型
        
        Args:
            path: 保存路径
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """
        加载智能体模型
        
        Args:
            path: 模型路径
        """
        pass
    
    def set_training_mode(self, training):
        """设置训练模式"""
        self.training = training
    
    def reset(self):
        """重置智能体状态"""
        pass


class RandomAgent(BaseAgent):
    """随机智能体，用于基线比较"""
    
    def __init__(self, config):
        super().__init__(config)
        self.action_space = config.get('action_space', 10)
        
    def act(self, observation):
        """随机选择动作"""
        return np.random.randint(0, self.action_space)
    
    def learn(self, experience):
        """随机智能体不学习"""
        pass
    
    def save(self, path):
        """随机智能体无需保存"""
        pass
    
    def load(self, path):
        """随机智能体无需加载"""
        pass


class GreedyAgent(BaseAgent):
    """贪心智能体，用于基线比较"""
    
    def __init__(self, config):
        super().__init__(config)
        self.action_space = config.get('action_space', 10)
        
    def act(self, observation):
        """选择贪心动作"""
        # 计算每个可能动作的即时奖励
        spins = observation[:, 0]  # 第一列是自旋状态
        graph = observation[:, 1:]  # 剩余列是邻接矩阵
        
        best_action = 0
        best_reward = -float('inf')
        
        for action in range(self.action_space):
            # 模拟翻转该节点
            if spins[action] == 1:  # 只能从1翻转到-1（不可逆模式）
                new_spins = spins.copy()
                new_spins[action] = -1
                
                # 计算新割值
                cut_value = 0
                for i in range(len(spins)):
                    for j in range(i+1, len(spins)):
                        if graph[i, j] > 0:
                            cut_value += (1 - new_spins[i] * new_spins[j]) * graph[i, j]
                
                reward = cut_value / 4
                
                if reward > best_reward:
                    best_reward = reward
                    best_action = action
        
        return best_action
    
    def learn(self, experience):
        """贪心智能体不学习"""
        pass
    
    def save(self, path):
        """贪心智能体无需保存"""
        pass
    
    def load(self, path):
        """贪心智能体无需加载"""
        pass