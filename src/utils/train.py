"""
训练工具类
提供通用的训练和评估功能
"""

import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json


class Trainer:
    """通用训练器类"""
    
    def __init__(self, config):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.save_frequency = config.get('save_frequency', 1000)
        self.log_frequency = config.get('log_frequency', 100)
        self.test_frequency = config.get('test_frequency', 500)
        self.device = config.get('device', 'cpu')
        
        # 创建保存目录
        self.model_dir = config.get('model_dir', 'models')
        self.log_dir = config.get('log_dir', 'logs')
        self.result_dir = config.get('result_dir', 'results')
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
    
    def train(self, graphs, create_env_fn, create_agent_fn, num_episodes, algorithm_name):
        """
        训练算法
        
        Args:
            graphs: 训练图列表
            create_env_fn: 创建环境的函数
            create_agent_fn: 创建智能体的函数
            num_episodes: 训练回合数
            algorithm_name: 算法名称
            
        Returns:
            results: 训练结果
        """
        # 创建环境和智能体
        env = create_env_fn(graphs[0])
        agent = create_agent_fn(env.get_observation_space(), env.get_action_space())
        
        # 训练统计
        episode_rewards = []
        episode_scores = []
        episode_best_scores = []
        episode_steps = []
        
        start_time = time.time()
        
        # 训练循环
        for episode in tqdm(range(num_episodes), desc=f"训练 {algorithm_name}"):
            # 随机选择一个图
            graph_idx = np.random.randint(0, len(graphs))
            graph = graphs[graph_idx]
            
            # 创建新环境
            env = create_env_fn(graph)
            
            # 运行一个回合
            episode_result = self._run_single_episode(env, agent, graph_idx)
            
            # 记录统计
            episode_rewards.append(episode_result['total_reward'])
            episode_scores.append(episode_result['final_score'])
            episode_best_scores.append(episode_result['best_score'])
            episode_steps.append(episode_result['steps'])
            
            # 定期保存模型
            if episode % self.save_frequency == 0:
                model_path = os.path.join(
                    self.model_dir, 
                    f"{algorithm_name}_episode_{episode}.pth"
                )
                agent.save(model_path)
            
            # 定期记录日志
            if episode % self.log_frequency == 0:
                self._log_progress(episode, episode_result, start_time)
        
        # 保存最终模型
        final_model_path = os.path.join(self.model_dir, f"{algorithm_name}_final.pth")
        agent.save(final_model_path)
        
        # 保存训练结果
        results = {
            'episode_rewards': episode_rewards,
            'episode_scores': episode_scores,
            'episode_best_scores': episode_best_scores,
            'episode_steps': episode_steps,
            'final_scores': episode_best_scores[-10:],  # 最后10个回合的最佳分数
            'mean_final_score': np.mean(episode_best_scores[-10:]),
            'best_score': max(episode_best_scores),
            'training_time': time.time() - start_time,
        }
        
        # 保存结果
        result_path = os.path.join(self.result_dir, f"{algorithm_name}_train_results.json")
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 绘制训练曲线
        self._plot_training_curves(results, algorithm_name)
        
        return results
    
    def evaluate(self, graphs, create_env_fn, create_agent_fn, model_path, algorithm_name):
        """
        评估算法
        
        Args:
            graphs: 测试图列表
            create_env_fn: 创建环境的函数
            create_agent_fn: 创建智能体的函数
            model_path: 模型路径
            algorithm_name: 算法名称
            
        Returns:
            results: 评估结果
        """
        # 创建环境和智能体
        env = create_env_fn(graphs[0])
        agent = create_agent_fn(env.get_observation_space(), env.get_action_space())
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            agent.load(model_path)
            print(f"已加载模型: {model_path}")
        
        # 设置为评估模式
        agent.set_training_mode(False)
        
        # 评估统计
        test_scores = []
        test_best_scores = []
        test_steps = []
        solutions = []
        
        start_time = time.time()
        
        # 评估循环
        for graph_idx, graph in enumerate(tqdm(graphs, desc=f"评估 {algorithm_name}")):
            # 创建环境
            env = create_env_fn(graph)
            
            # 运行一个回合
            episode_result = self._run_single_episode(env, agent, graph_idx)
            
            # 记录统计
            test_scores.append(episode_result['final_score'])
            test_best_scores.append(episode_result['best_score'])
            test_steps.append(episode_result['steps'])
            solutions.append(episode_result['best_solution'])
        
        # 保存评估结果
        results = {
            'test_scores': test_scores,
            'test_best_scores': test_best_scores,
            'test_steps': test_steps,
            'solutions': solutions,
            'mean_score': np.mean(test_best_scores),
            'best_score': max(test_best_scores),
            'std_score': np.std(test_best_scores),
            'evaluation_time': time.time() - start_time,
        }
        
        # 保存结果
        result_path = os.path.join(self.result_dir, f"{algorithm_name}_eval_results.json")
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 打印结果
        print(f"\n评估结果:")
        print(f"平均分数: {results['mean_score']:.2f}")
        print(f"最佳分数: {results['best_score']:.2f}")
        print(f"标准差: {results['std_score']:.2f}")
        print(f"评估时间: {results['evaluation_time']:.2f}秒")
        
        return results
    
    def _run_single_episode(self, env, agent, graph_id):
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
    
    def _log_progress(self, episode, episode_result, start_time):
        """记录训练进度"""
        elapsed_time = time.time() - start_time
        print(f"回合 {episode}: "
              f"分数={episode_result['best_score']:.2f}, "
              f"步数={episode_result['steps']}, "
              f"时间={elapsed_time:.1f}s")
    
    def _plot_training_curves(self, results, algorithm_name):
        """绘制训练曲线"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 分数曲线
            plt.subplot(2, 2, 1)
            plt.plot(results['episode_scores'])
            plt.title('回合分数')
            plt.xlabel('回合')
            plt.ylabel('分数')
            plt.grid(True)
            
            # 最佳分数曲线
            plt.subplot(2, 2, 2)
            plt.plot(results['episode_best_scores'])
            plt.title('最佳分数')
            plt.xlabel('回合')
            plt.ylabel('最佳分数')
            plt.grid(True)
            
            # 步数曲线
            plt.subplot(2, 2, 3)
            plt.plot(results['episode_steps'])
            plt.title('回合步数')
            plt.xlabel('回合')
            plt.ylabel('步数')
            plt.grid(True)
            
            # 奖励曲线
            plt.subplot(2, 2, 4)
            plt.plot(results['episode_rewards'])
            plt.title('回合奖励')
            plt.xlabel('回合')
            plt.ylabel('奖励')
            plt.grid(True)
            
            plt.tight_layout()
            
            # 保存图像
            plot_path = os.path.join(self.result_dir, f"{algorithm_name}_training_curves.png")
            plt.savefig(plot_path)
            plt.close()
            
            print(f"训练曲线已保存到: {plot_path}")
            
        except Exception as e:
            print(f"绘制训练曲线时出错: {e}")