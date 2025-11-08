"""
S2V算法示例脚本
演示如何使用S2V算法解决最大割问题
"""

import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.algorithms.s2v import S2VAlgorithm, create_s2v_config
from src.algorithms.eco import ECOAlgorithm, create_eco_config
from src.utils.graph import generate_random_graph, generate_graph_dataset
from config import DEFAULT_CONFIG, AlgorithmType, GraphType


def main():
    """主函数"""
    print("=" * 50)
    print("S2V算法示例 - 最大割问题")
    print("=" * 50)
    
    # 创建S2V配置
    config = create_s2v_config(DEFAULT_CONFIG)
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
    print(f"  算法: S2V")
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
    
    # 创建S2V算法
    s2v = S2VAlgorithm(config)
    
    # 训练
    print("\n开始训练...")
    train_results = s2v.train(train_graphs, num_episodes=100)
    
    # 评估
    print("\n开始评估...")
    model_path = os.path.join(config['model_dir'], 'S2V_final.pth')
    eval_results = s2v.evaluate(test_graphs, model_path)
    
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
    print("S2V算法示例完成!")
    print("结果保存在 models/ 和 results/ 目录中")


def compare_eco_s2v():
    """比较ECO和S2V算法"""
    print("\n比较ECO和S2V算法...")
    
    # 生成测试图
    test_graphs = [
        generate_random_graph(20, 'BA', seed=i)
        for i in range(5)
    ]
    
    # 测试ECO
    eco_config = create_eco_config(DEFAULT_CONFIG)
    eco_config.update({
        'num_steps': 2000,  # 较少步数用于快速比较
        'device': 'cpu',
    })
    
    eco = ECOAlgorithm(eco_config)
    eco_results = eco.evaluate(test_graphs)
    
    # 测试S2V
    s2v_config = create_s2v_config(DEFAULT_CONFIG)
    s2v_config.update({
        'num_steps': 2000,  # 较少步数用于快速比较
        'device': 'cpu',
    })
    
    s2v = S2VAlgorithm(s2v_config)
    s2v_results = s2v.evaluate(test_graphs)
    
    # 比较结果
    print("\n比较结果:")
    print(f"ECO平均分数: {eco_results['mean_score']:.2f} ± {eco_results['std_score']:.2f}")
    print(f"S2V平均分数: {s2v_results['mean_score']:.2f} ± {s2v_results['std_score']:.2f}")
    
    if eco_results['mean_score'] > s2v_results['mean_score']:
        print("ECO算法表现更好")
    else:
        print("S2V算法表现更好")


def analyze_reward_signals():
    """分析不同奖励信号的影响"""
    print("\n分析不同奖励信号的影响...")
    
    # 生成测试图
    test_graph = generate_random_graph(20, 'BA', seed=42)
    
    # 测试BLS奖励
    eco_config = create_eco_config(DEFAULT_CONFIG)
    eco_config.update({
        'num_steps': 1000,
        'device': 'cpu',
    })
    
    eco = ECOAlgorithm(eco_config)
    eco_results = eco.train([test_graph], num_episodes=20)
    
    # 测试DENSE奖励
    s2v_config = create_s2v_config(DEFAULT_CONFIG)
    s2v_config.update({
        'num_steps': 1000,
        'device': 'cpu',
    })
    
    s2v = S2VAlgorithm(s2v_config)
    s2v_results = s2v.train([test_graph], num_episodes=20)
    
    # 分析结果
    print("\n奖励信号分析:")
    print(f"BLS奖励 (ECO): 平均分数 {eco_results['mean_final_score']:.2f}")
    print(f"DENSE奖励 (S2V): 平均分数 {s2v_results['mean_final_score']:.2f}")
    
    # 分析学习曲线
    eco_scores = eco_results['episode_scores']
    s2v_scores = s2v_results['episode_scores']
    
    print(f"\n学习稳定性:")
    print(f"ECO最后10回合标准差: {np.std(eco_scores[-10:]):.2f}")
    print(f"S2V最后10回合标准差: {np.std(s2v_scores[-10:]):.2f}")


def test_different_graph_sizes():
    """测试不同图大小的性能"""
    print("\n测试不同图大小的性能...")
    
    node_sizes = [10, 20, 50, 100]
    eco_scores = []
    s2v_scores = []
    
    for n_nodes in node_sizes:
        print(f"\n测试 {n_nodes} 节点的图...")
        
        # 生成图
        test_graph = generate_random_graph(n_nodes, 'BA', seed=42)
        
        # 测试ECO
        eco_config = create_eco_config(DEFAULT_CONFIG)
        eco_config.update({
            'num_steps': 1000,
            'device': 'cpu',
        })
        
        eco = ECOAlgorithm(eco_config)
        eco_result = eco.evaluate([test_graph])
        eco_scores.append(eco_result['mean_score'])
        
        # 测试S2V
        s2v_config = create_s2v_config(DEFAULT_CONFIG)
        s2v_config.update({
            'num_steps': 1000,
            'device': 'cpu',
        })
        
        s2v = S2VAlgorithm(s2v_config)
        s2v_result = s2v.evaluate([test_graph])
        s2v_scores.append(s2v_result['mean_score'])
    
    # 打印结果
    print("\n图大小分析:")
    print("节点数 | ECO分数 | S2V分数")
    print("-" * 30)
    for i, n_nodes in enumerate(node_sizes):
        print(f"{n_nodes:6d} | {eco_scores[i]:8.2f} | {s2v_scores[i]:8.2f}")


if __name__ == "__main__":
    # 运行主示例
    main()
    
    # 可选：比较ECO和S2V
    # compare_eco_s2v()
    
    # 可选：分析奖励信号
    # analyze_reward_signals()
    
    # 可选：测试不同图大小
    # test_different_graph_sizes()