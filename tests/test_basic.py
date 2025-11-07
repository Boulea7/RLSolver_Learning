"""
基础测试脚本
测试环境和智能体的基本功能
"""

import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environments.maxcut import MaxCutEnvironment
from src.agents.base import RandomAgent, GreedyAgent
from src.agents.dqn import DQNAgent
from src.utils.graph import generate_random_graph
from config import DEFAULT_CONFIG


def test_environment():
    """测试环境功能"""
    print("=" * 50)
    print("测试环境功能")
    print("=" * 50)
    
    # 生成测试图
    graph = generate_random_graph(10, 'BA', seed=42)
    
    # 创建环境
    env_config = DEFAULT_CONFIG.copy()
    env = MaxCutEnvironment(graph, env_config)
    
    print(f"图信息: {env.get_problem_info()}")
    
    # 测试重置
    print("\n测试环境重置...")
    observation = env.reset()
    print(f"观察空间: {env.get_observation_space()}")
    print(f"动作空间: {env.get_action_space()}")
    print(f"初始分数: {env.calculate_score()}")
    
    # 测试步进
    print("\n测试环境步进...")
    total_reward = 0
    for step in range(5):
        action = step % env.get_action_space()  # 简单策略：轮流选择动作
        observation, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"步骤 {step+1}: 动作={action}, 奖励={reward:.2f}, "
              f"分数={env.calculate_score():.2f}, 完成={done}")
        
        if done:
            break
    
    print(f"\n总奖励: {total_reward:.2f}")
    print(f"最佳分数: {env.get_best_score():.2f}")
    
    # 获取解
    solution = env.get_solution()
    print(f"\n解信息:")
    print(f"  最终分数: {solution['cut_value']:.2f}")
    print(f"  最佳分数: {solution['best_cut_value']:.2f}")
    print(f"  解: {solution['best_solution']}")


def test_agents():
    """测试智能体功能"""
    print("\n" + "=" * 50)
    print("测试智能体功能")
    print("=" * 50)
    
    # 生成测试图
    graph = generate_random_graph(10, 'BA', seed=42)
    env_config = DEFAULT_CONFIG.copy()
    env = MaxCutEnvironment(graph, env_config)
    
    # 测试随机智能体
    print("\n测试随机智能体...")
    random_agent = RandomAgent({'action_space': env.get_action_space()})
    
    total_rewards = []
    for episode in range(5):
        observation = env.reset()
        random_agent.reset()
        
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 20:
            action = random_agent.act(observation)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        total_rewards.append(episode_reward)
        print(f"回合 {episode+1}: 奖励={episode_reward:.2f}, "
              f"分数={env.get_best_score():.2f}, 步数={steps}")
    
    print(f"随机智能体平均奖励: {np.mean(total_rewards):.2f}")
    
    # 测试贪心智能体
    print("\n测试贪心智能体...")
    greedy_agent = GreedyAgent({'action_space': env.get_action_space()})
    
    total_rewards = []
    for episode in range(5):
        observation = env.reset()
        greedy_agent.reset()
        
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 20:
            action = greedy_agent.act(observation)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        total_rewards.append(episode_reward)
        print(f"回合 {episode+1}: 奖励={episode_reward:.2f}, "
              f"分数={env.get_best_score():.2f}, 步数={steps}")
    
    print(f"贪心智能体平均奖励: {np.mean(total_rewards):.2f}")


def test_dqn_agent():
    """测试DQN智能体"""
    print("\n" + "=" * 50)
    print("测试DQN智能体")
    print("=" * 50)
    
    # 生成测试图
    graph = generate_random_graph(10, 'BA', seed=42)
    env_config = DEFAULT_CONFIG.copy()
    env = MaxCutEnvironment(graph, env_config)
    
    # 创建DQN智能体
    agent_config = DEFAULT_CONFIG.copy()
    agent_config.update({
        'input_dim': np.prod(env.get_observation_space()),
        'action_space': env.get_action_space(),
        'replay_buffer_size': 1000,
        'replay_start_size': 100,
        'num_steps': 500,
        'device': 'cpu',
    })
    
    dqn_agent = DQNAgent(agent_config)
    
    print("开始训练DQN智能体...")
    total_rewards = []
    
    for episode in range(10):
        observation = env.reset()
        dqn_agent.reset()
        
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 20:
            action = dqn_agent.act(observation)
            observation, reward, done, info = env.step(action)
            
            # 智能体学习
            dqn_agent.learn((observation, action, reward, observation, done))
            
            episode_reward += reward
            steps += 1
        
        total_rewards.append(episode_reward)
        
        if episode % 2 == 0:
            print(f"回合 {episode+1}: 奖励={episode_reward:.2f}, "
                  f"分数={env.get_best_score():.2f}, 步数={steps}, "
                  f"ε={dqn_agent.epsilon:.3f}")
    
    print(f"DQN智能体平均奖励: {np.mean(total_rewards):.2f}")
    print(f"最终探索率: {dqn_agent.epsilon:.3f}")


def test_graph_utilities():
    """测试图工具功能"""
    print("\n" + "=" * 50)
    print("测试图工具功能")
    print("=" * 50)
    
    from src.utils.graph import (
        calculate_graph_properties, 
        visualize_graph, 
        save_graph_to_file,
        load_graph_from_file
    )
    
    # 生成不同类型的图
    print("\n生成不同类型的图...")
    
    ba_graph = generate_random_graph(20, 'BA', seed=42)
    er_graph = generate_random_graph(20, 'ER', seed=42)
    
    # 计算属性
    ba_props = calculate_graph_properties(ba_graph)
    er_props = calculate_graph_properties(er_graph)
    
    print("\nBA图属性:")
    for key, value in ba_props.items():
        print(f"  {key}: {value}")
    
    print("\nER图属性:")
    for key, value in er_props.items():
        print(f"  {key}: {value}")
    
    # 保存图
    print("\n保存图到文件...")
    save_graph_to_file(ba_graph, "test_ba_graph.txt")
    save_graph_to_file(er_graph, "test_er_graph.txt")
    print("图已保存到 test_ba_graph.txt 和 test_er_graph.txt")
    
    # 测试加载图
    print("\n测试加载图...")
    loaded_graph = load_graph_from_file("test_ba_graph.txt")
    print(f"加载的图形状: {loaded_graph.shape}")
    print(f"图是否相同: {np.allclose(ba_graph, loaded_graph)}")
    
    # 可视化图（如果matplotlib可用）
    try:
        print("\n可视化图...")
        visualize_graph(ba_graph, title="BA Graph (20 nodes)")
        print("BA图可视化已显示")
    except Exception as e:
        print(f"可视化失败: {e}")


def run_all_tests():
    """运行所有测试"""
    print("开始运行所有基础测试...")
    
    try:
        test_environment()
        print("✓ 环境测试通过")
    except Exception as e:
        print(f"✗ 环境测试失败: {e}")
    
    try:
        test_agents()
        print("✓ 智能体测试通过")
    except Exception as e:
        print(f"✗ 智能体测试失败: {e}")
    
    try:
        test_dqn_agent()
        print("✓ DQN智能体测试通过")
    except Exception as e:
        print(f"✗ DQN智能体测试失败: {e}")
    
    try:
        test_graph_utilities()
        print("✓ 图工具测试通过")
    except Exception as e:
        print(f"✗ 图工具测试失败: {e}")
    
    print("\n所有测试完成!")


if __name__ == "__main__":
    run_all_tests()