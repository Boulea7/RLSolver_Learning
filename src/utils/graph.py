"""
图处理工具
提供图的生成、加载和处理功能
"""

import numpy as np
import networkx as nx
import os


def generate_random_graph(n_nodes, graph_type='BA', p=0.15, m=4, seed=None):
    """
    生成随机图
    
    Args:
        n_nodes: 节点数
        graph_type: 图类型 ('BA' 或 'ER')
        p: ER图的连接概率
        m: BA图的每个新节点连接的边数
        seed: 随机种子
        
    Returns:
        graph: 邻接矩阵
    """
    if seed is not None:
        np.random.seed(seed)
    
    if graph_type == 'BA':
        # Barabási-Albert图
        G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    elif graph_type == 'ER':
        # Erdős-Rényi图
        G = nx.erdos_renyi_graph(n_nodes, p, seed=seed)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    # 设置权重为1
    for u, v in G.edges():
        G[u][v]['weight'] = 1
    
    return nx.to_numpy_array(G)


def load_graph_from_file(file_path):
    """
    从文件加载图
    
    Args:
        file_path: 文件路径
        
    Returns:
        graph: 邻接矩阵
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 第一行包含节点数和边数
    n_nodes, n_edges = map(int, lines[0].split())
    
    # 初始化邻接矩阵
    graph = np.zeros((n_nodes, n_nodes))
    
    # 读取边
    for line in lines[1:]:
        if line.strip():
            parts = line.split()
            if len(parts) >= 3:
                u, v, weight = map(float, parts[:3])
            else:
                u, v = map(float, parts[:2])
                weight = 1.0
            
            u, v = int(u) - 1, int(v) - 1  # 转换为0-based索引
            graph[u, v] = graph[v, u] = weight
    
    return graph


def save_graph_to_file(graph, file_path):
    """
    保存图到文件
    
    Args:
        graph: 邻接矩阵
        file_path: 保存路径
    """
    n_nodes = graph.shape[0]
    
    # 计算边数
    n_edges = np.sum(graph > 0) // 2
    
    with open(file_path, 'w') as f:
        # 写入节点数和边数
        f.write(f"{n_nodes} {n_edges}\n")
        
        # 写入边
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if graph[i, j] > 0:
                    f.write(f"{i+1} {j+1} {graph[i, j]}\n")


def calculate_graph_properties(graph):
    """
    计算图的属性
    
    Args:
        graph: 邻接矩阵
        
    Returns:
        properties: 图属性字典
    """
    n_nodes = graph.shape[0]
    n_edges = np.sum(graph > 0) // 2
    
    # 度分布
    degrees = np.sum(graph > 0, axis=1)
    
    # 聚类系数
    G = nx.from_numpy_array(graph)
    clustering = nx.average_clustering(G)
    
    # 连通性
    is_connected = nx.is_connected(G)
    
    # 密度
    density = 2 * n_edges / (n_nodes * (n_nodes - 1))
    
    return {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'density': density,
        'avg_degree': np.mean(degrees),
        'max_degree': np.max(degrees),
        'min_degree': np.min(degrees),
        'clustering_coefficient': clustering,
        'is_connected': is_connected,
    }


def visualize_graph(graph, node_colors=None, layout='spring', save_path=None, title="Graph"):
    """
    可视化图
    
    Args:
        graph: 邻接矩阵
        node_colors: 节点颜色列表
        layout: 布局类型 ('spring', 'circular', 'random')
        save_path: 保存路径
        title: 图标题
    """
    try:
        import matplotlib.pyplot as plt
        
        # 创建NetworkX图
        G = nx.from_numpy_array(graph)
        
        # 设置布局
        if layout == 'spring':
            pos = nx.spring_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'random':
            pos = nx.random_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # 设置节点颜色
        if node_colors is None:
            node_colors = 'lightblue'
        
        # 绘制图
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, node_color=node_colors, with_labels=True, 
                node_size=500, font_size=10, font_weight='bold')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("Matplotlib not available for visualization")


def generate_graph_dataset(n_nodes_list, graph_type='BA', num_graphs_per_size=10, 
                       output_dir='data', seed=None):
    """
    生成图数据集
    
    Args:
        n_nodes_list: 节点数列表
        graph_type: 图类型
        num_graphs_per_size: 每种大小的图数量
        output_dir: 输出目录
        seed: 随机种子
    """
    if seed is not None:
        np.random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for n_nodes in n_nodes_list:
        for i in range(num_graphs_per_size):
            # 生成图
            graph = generate_random_graph(n_nodes, graph_type, seed=seed+i if seed else None)
            
            # 保存图
            filename = f"{graph_type}_{n_nodes}_ID{i}.txt"
            filepath = os.path.join(output_dir, filename)
            save_graph_to_file(graph, filepath)
    
    print(f"已生成 {len(n_nodes_list) * num_graphs_per_size} 个图到 {output_dir}")


def compare_graphs(graph1, graph2):
    """
    比较两个图
    
    Args:
        graph1: 第一个图的邻接矩阵
        graph2: 第二个图的邻接矩阵
        
    Returns:
        comparison: 比较结果
    """
    # 基本属性
    props1 = calculate_graph_properties(graph1)
    props2 = calculate_graph_properties(graph2)
    
    # 结构相似性
    if graph1.shape == graph2.shape:
        similarity = np.sum(graph1 == graph2) / np.prod(graph1.shape)
    else:
        similarity = 0.0
    
    return {
        'graph1_properties': props1,
        'graph2_properties': props2,
        'structural_similarity': similarity,
        'same_shape': graph1.shape == graph2.shape,
    }


if __name__ == "__main__":
    """测试图工具"""
    # 生成随机图
    graph = generate_random_graph(20, 'BA', seed=42)
    print("生成的图属性:")
    props = calculate_graph_properties(graph)
    for key, value in props.items():
        print(f"  {key}: {value}")
    
    # 可视化图
    visualize_graph(graph, title="BA Graph (20 nodes)")
    
    # 保存图
    save_graph_to_file(graph, "test_graph.txt")
    print("图已保存到 test_graph.txt")