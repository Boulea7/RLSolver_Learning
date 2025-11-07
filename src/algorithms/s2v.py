"""
S2V (Structure2Vector) 算法的简化实现
基于原始RLSolver的S2V算法，使用密集奖励模式和不可逆自旋系统
"""

import numpy as np
import torch
from tqdm import tqdm

from ..environments.maxcut import MaxCutEnvironment
from ..agents.dqn import DQNAgent
from ..utils.train import Trainer


class S2VAlgorithm:
    """S2V算法的简化实现"""
    
    def __init__(self, config):
        """
        初始化S2V算法
        
        Args:
            config: 算法配置
        """
        self.config = config
        self.algorithm_name = 'S2V'
        
        # S2V特定配置
        self.reversible_spins = False  # 不可逆自旋系统
        self.reward_signal = 'DENSE'  # 密集奖励
        
    def create_environment(self, graph):
        """创建S2V环境"""
        env_config = self.config.copy()
        env_config.update({
            'reversible_spins': self.reversible_spins,
            'reward_signal': self.reward_signal,
        })
        return MaxCutEnvironment(graph, env_config)
    
    def create_agent(self, observation_space, action_space):
        """创建DQN智能体"""
        agent_config = self.config.copy()
        agent_config.update({
            'input_dim': np.prod(observation_space),
            'action_space': action_space,
        })
        return DQNAgent(agent_config)
    
    def train(self, graphs, num_episodes=1000):
        """
        训练S2V算法
        
        Args:
            graphs: 训练图列表
            num_episodes: 训练回合数
        """
        print(f"开始训练 {self.algorithm_name} 算法")
        print(f"训练图数量: {len(graphs)}")
        print(f"训练回合数: {num_episodes}")
        
        # 创建训练器
        trainer = Trainer(self.config)
        
        # 训练循环
        results = trainer.train(
            graphs=graphs,
            create_env_fn=self.create_environment,
            create_agent_fn=self.create_agent,
            num_episodes=num_episodes,
            algorithm_name=self.algorithm_name
        )
        
        return results
    
    def evaluate(self, graphs, model_path=None):
        """
        评估S2V算法
        
        Args:
            graphs: 测试图列表
            model_path: 模型路径
        """
        print(f"开始评估 {self.algorithm_name} 算法")
        print(f"测试图数量: {len(graphs)}")
        
        # 创建训练器
        trainer = Trainer(self.config)
        
        # 评估
        results = trainer.evaluate(
            graphs=graphs,
            create_env_fn=self.create_environment,
            create_agent_fn=self.create_agent,
            model_path=model_path,
            algorithm_name=self.algorithm_name
        )
        
        return results
    
    def run_single_episode(self, env, agent, graph_id=0):
        """
        运行单个回合
        
        Args:
            env: 环境
            agent: 智能体
            graph_id: 图ID
            
        Returns:
            episode_result: 回合结果
        """
        # 重置环境和智能体
        observation = env.reset()
        agent.reset()
        
        total_reward = 0
        done = False
        step_count = 0
        
        while not done:
            # 选择动作
            action = agent.act(observation)
            
            # 执行动作
            next_observation, reward, done, info = env.step(action)
            
            # 智能体学习
            if agent.training:
                agent.learn((observation, action, reward, next_observation, done))
            
            total_reward += reward
            observation = next_observation
            step_count += 1
        
        # 获取结果
        solution = env.get_solution()
        
        return {
            'graph_id': graph_id,
            'total_reward': total_reward,
            'steps': step_count,
            'final_score': solution['cut_value'],
            'best_score': solution['best_cut_value'],
            'solution': solution['cut_set'],
            'best_solution': solution['best_cut_set'],
        }
    
    def get_algorithm_info(self):
        """获取算法信息"""
        return {
            'name': self.algorithm_name,
            'reversible_spins': self.reversible_spins,
            'reward_signal': self.reward_signal,
            'description': 'S2V算法使用密集奖励和不可逆自旋系统',
        }


def create_s2v_config(base_config):
    """
    创建S2V特定的配置
    
    Args:
        base_config: 基础配置
        
    Returns:
        s2v_config: S2V配置
    """
    s2v_config = base_config.copy()
    s2v_config.update({
        'reversible_spins': False,
        'reward_signal': 'DENSE',
        'gamma': 1.0,  # S2V通常使用gamma=1
        'max_steps_per_episode': 20,  # 节点数（不可逆模式）
    })
    return s2v_config


if __name__ == "__main__":
    """测试S2V算法"""
    from ...config import DEFAULT_CONFIG, AlgorithmType, GraphType
    
    # 创建配置
    config = create_s2v_config(DEFAULT_CONFIG)
    config.update({
        'num_train_nodes': 20,
        'num_steps': 1000,
        'device': 'cpu',
    })
    
    # 创建算法
    s2v = S2VAlgorithm(config)
    
    # 生成随机图
    from ..utils.graph import generate_random_graph
    graphs = [
        generate_random_graph(20, 'BA')
        for _ in range(10)
    ]
    
    # 训练
    results = s2v.train(graphs, num_episodes=100)
    print("训练完成!")
    print(f"最终得分: {results['final_scores']}")