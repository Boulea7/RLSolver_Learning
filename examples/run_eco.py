"""
ECO算法示例脚本
演示如何使用ECO算法解决最大割问题
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.algorithms.eco import ECOAlgorithm, create_eco_config
from src.utils.graph import generate_random_graph, generate_graph_dataset
from config import DEFAULT_CONFIG, AlgorithmType, GraphType


def main():
    """主函数"""
    print("=" * 50)
    print("ECO算法示例 - 最大割问题")
    print("=" * 50)
    
    # 创建ECO配置
    config = create_eco_config(DEFAULT_CONFIG)
    config.update({
        'num_train_nodes': 20,
        'num_train_envs': 32,
        'num_steps': 5000,
        'device': 'cpu',  # 使用CPU，如果有GPU可以改为'cuda'
        'save_frequency': 1000,
        'log_frequency': 100,
        'test_frequency': 500,
    })
    
    # 打印配置
    print("\n配置信息:")
    print(f"  算法: ECO")
    print(f"  节点数: {config['num_train_nodes']}")
    print(f"  训练步数: {config['num_steps']}")
    print(f"  设备: {config['device']}")
    print(f"  可逆自旋: {config['reversible_spins']}")
    print(f"  奖励信号: {config['reward_signal']}")
    
    # 生成训练图
    print("\n生成训练图...")
    train_graphs = [
        generate_random_graph(config['num_train_nodes'], 'BA', seed=i)
        for i in range(10)
    ]
    print(f"已生成 {len(train_graphs)} 个训练图")
    
    # 生成测试图
    print("\n生成测试图...")
    test_graphs = [
        generate_random_graph(50, 'BA', seed=i+100)  # 更大的测试图
        for i in range(5)
    ]
    print(f"已生成 {len(test_graphs)} 个测试图")
    
    # 创建ECO算法
    eco = ECOAlgorithm(config)
    
    # 训练
    print("\n开始训练...")
    train_results = eco.train(train_graphs, num_episodes=100)
    
    # 评估
    print("\n开始评估...")
    model_path = os.path.join(config['model_dir'], 'ECO_final.pth')
    eval_results = eco.evaluate(test_graphs, model_path)
    
    # 打印结果
    print("\n" + "=" * 50)
    print("训练结果:")
    print(f"  最佳分数: {train_results['best_score']:.2f}")
    print(f"  平均分数: {train_results['mean_final_score']:.2f}")
    print(f"  训练时间: {train_results['training_time']:.2f}秒")
    
    print("\n评估结果:")
    print(f"  最佳分数: {eval_results['best_score']:.2f}")
    print(f"  平均分数: {eval_results['mean_score']:.2f}")
    print(f"  标准差: {eval_results['std_score']:.2f}")
    print(f"  评估时间: {eval_results['evaluation_time']:.2f}秒")
    
    print("\n" + "=" * 50)
    print("ECO算法示例完成!")
    print("结果保存在 models/ 和 results/ 目录中")


def generate_dataset_example():
    """生成数据集示例"""
    print("\n生成数据集示例...")
    
    # 生成不同大小的图
    n_nodes_list = [20, 50, 100]
    generate_graph_dataset(
        n_nodes_list=n_nodes_list,
        graph_type='BA',
        num_graphs_per_size=10,
        output_dir='data',
        seed=42
    )
    
    print("数据集已生成到 data/ 目录")


def compare_with_baseline():
    """与基线方法比较"""
    print("\n与基线方法比较...")
    
    from src.agents.base import RandomAgent, GreedyAgent
    from src.environments.maxcut import MaxCutEnvironment
    
    # 生成测试图
    test_graph = generate_random_graph(20, 'BA', seed=42)
    env = MaxCutEnvironment(test_graph, config)
    
    # 测试随机智能体
    random_agent = RandomAgent({'action_space': 20})
    random_result = run_baseline_test(env, random_agent, "随机智能体")
    
    # 测试贪心智能体
    greedy_agent = GreedyAgent({'action_space': 20})
    greedy_result = run_baseline_test(env, greedy_agent, "贪心智能体")
    
    print(f"随机智能体分数: {random_result:.2f}")
    print(f"贪心智能体分数: {greedy_result:.2f}")


def run_baseline_test(env, agent, agent_name):
    """运行基线测试"""
    total_scores = []
    
    for _ in range(10):  # 运行10次取平均
        observation = env.reset()
        agent.reset()
        
        done = False
        while not done:
            action = agent.act(observation)
            _, _, done, _ = env.step(action)
        
        total_scores.append(env.get_best_score())
    
    return np.mean(total_scores)


if __name__ == "__main__":
    # 运行主示例
    main()
    
    # 可选：生成数据集
    # generate_dataset_example()
    
    # 可选：与基线比较
    # compare_with_baseline()