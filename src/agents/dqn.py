"""
简化的深度Q网络（DQN）智能体实现
基于原始RLSolver的DQN简化而来
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

from .base import BaseAgent


class QNetwork(nn.Module):
    """简化的Q网络"""
    
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=3):
        """
        初始化Q网络
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度（动作空间大小）
            hidden_dim: 隐藏层维度
            num_layers: 网络层数
        """
        super(QNetwork, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        # 构建隐藏层
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """前向传播"""
        return self.network(x)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity):
        """
        初始化回放缓冲区
        
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """采样经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        """返回缓冲区大小"""
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """简化的DQN智能体"""
    
    def __init__(self, config):
        """
        初始化DQN智能体
        
        Args:
            config: 配置字典，包含以下键：
                - input_dim: 输入维度
                - action_space: 动作空间大小
                - learning_rate: 学习率
                - gamma: 折扣因子
                - replay_buffer_size: 经验回放缓冲区大小
                - minibatch_size: 小批量大小
                - update_frequency: 更新频率
                - update_target_frequency: 目标网络更新频率
                - epsilon_start: 初始探索率
                - epsilon_end: 最终探索率
                - epsilon_decay_steps: 探索率衰减步数
                - device: 计算设备
        """
        super().__init__(config)
        
        # 基本参数
        self.input_dim = config.get('input_dim')
        self.action_space = config.get('action_space')
        self.device = torch.device(config.get('device', 'cpu'))
        
        # 网络参数
        self.hidden_dim = config.get('network_features', 64)
        self.num_layers = config.get('network_layers', 3)
        
        # 训练参数
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.gamma = config.get('gamma', 0.95)
        self.minibatch_size = config.get('minibatch_size', 32)
        self.update_frequency = config.get('update_frequency', 32)
        self.update_target_frequency = config.get('update_target_frequency', 1000)
        
        # 探索参数
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.05)
        self.epsilon_decay_steps = config.get('epsilon_decay_steps', 5000)
        
        # 创建网络
        self.q_network = QNetwork(
            self.input_dim, 
            self.action_space, 
            self.hidden_dim, 
            self.num_layers
        ).to(self.device)
        
        self.target_network = QNetwork(
            self.input_dim, 
            self.action_space, 
            self.hidden_dim, 
            self.num_layers
        ).to(self.device)
        
        # 初始化目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(config.get('replay_buffer_size', 10000))
        self.replay_start_size = config.get('replay_start_size', 1000)
        
        # 训练状态
        self.step_count = 0
        self.update_count = 0
        
    def act(self, observation):
        """
        根据观察选择动作
        
        Args:
            observation: 当前观察，形状为 (n_nodes, features)
            
        Returns:
            action: 选择的动作
        """
        # 转换为张量
        state = torch.FloatTensor(observation.flatten()).unsqueeze(0).to(self.device)
        
        # ε-贪婪策略
        if self.training and random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        
        # 贪心选择
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def learn(self, experience):
        """
        从经验中学习
        
        Args:
            experience: 经验元组 (state, action, reward, next_state, done)
        """
        # 添加到回放缓冲区
        self.replay_buffer.push(*experience)
        self.step_count += 1
        
        # 更新探索率
        self._update_epsilon()
        
        # 检查是否可以开始训练
        if len(self.replay_buffer) < self.replay_start_size:
            return
        
        # 定期更新网络
        if self.step_count % self.update_frequency == 0:
            self._update_network()
        
        # 定期更新目标网络
        if self.step_count % self.update_target_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _update_network(self):
        """更新Q网络"""
        # 采样小批量
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.minibatch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_count += 1
        
        return loss.item()
    
    def _update_epsilon(self):
        """更新探索率"""
        if self.step_count <= self.epsilon_decay_steps:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (
                self.step_count / self.epsilon_decay_steps
            )
        else:
            self.epsilon = self.epsilon_end
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
    
    def reset(self):
        """重置智能体状态"""
        pass  # DQN智能体无需重置内部状态