"""
最大割问题环境实现
基于自旋系统，专门针对最大割问题
"""

import numpy as np
import networkx as nx
from .base import SpinSystemEnvironment


class MaxCutEnvironment(SpinSystemEnvironment):
    """最大割问题环境"""
    
    def __init__(self, graph, config):
        """
        初始化最大割环境
        
        Args:
            graph: 邻接矩阵或NetworkX图对象
            config: 环境配置
        """
        # 转换为邻接矩阵
        if isinstance(graph, nx.Graph):
            graph = nx.to_numpy_array(graph)
        
        super().__init__(graph, config)
        
        # 最大割特定配置
        self.problem_type = 'maxcut'
        
    def get_problem_info(self):
        """获取问题信息"""
        return {
            'problem_type': self.problem_type,
            'n_nodes': self.n_nodes,
            'n_edges': np.sum(self.graph > 0) // 2,
            'graph_density': np.sum(self.graph > 0) / (self.n_nodes * (self.n_nodes - 1)),
            'best_possible_score': self._calculate_upper_bound(),
        }
    
    def _calculate_upper_bound(self):
        """计算理论上界（所有边的权重和）"""
        return np.sum(self.graph) / 2
    
    def get_solution(self):
        """获取当前解"""
        # 将自旋转换为割集：1表示在集合A中，-1表示在集合B中
        cut_set = (self.spins > 0).astype(int)
        return {
            'cut_set': cut_set,
            'cut_value': self.current_score,
            'best_cut_set': (self.best_spins > 0).astype(int),
            'best_cut_value': self.best_score,
        }
    
    @staticmethod
    def generate_random_graph(n_nodes, graph_type='BA', p=0.15, m=4):
        """
        生成随机图
        
        Args:
            n_nodes: 节点数
            graph_type: 图类型 ('BA' 或 'ER')
            p: ER图的连接概率
            m: BA图的每个新节点连接的边数
            
        Returns:
            graph: 邻接矩阵
        """
        if graph_type == 'BA':
            # Barabási-Albert图
            G = nx.barabasi_albert_graph(n_nodes, m)
            # 确保是无向图且有权重
            G = nx.Graph()
            G.add_nodes_from(range(n_nodes))
            for i in range(n_nodes):
                for j in range(i+1, min(i+m+1, n_nodes)):
                    G.add_edge(i, j, weight=1)
        elif graph_type == 'ER':
            # Erdős-Rényi图
            G = nx.erdos_renyi_graph(n_nodes, p)
            # 设置权重为1
            for u, v in G.edges():
                G[u][v]['weight'] = 1
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        
        return nx.to_numpy_array(G)
    
    @staticmethod
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
                u, v, weight = map(float, line.split())
                u, v = int(u) - 1, int(v) - 1  # 转换为0-based索引
                graph[u, v] = graph[v, u] = weight
        
        return graph
    
    def visualize_solution(self, save_path=None):
        """可视化当前解"""
        try:
            import matplotlib.pyplot as plt
            
            # 创建NetworkX图
            G = nx.from_numpy_array(self.graph)
            
            # 设置节点颜色：根据割集
            colors = ['red' if spin > 0 else 'blue' for spin in self.spins]
            
            # 绘制图
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, node_color=colors, with_labels=True, node_size=500)
            plt.title(f"MaxCut Solution (Value: {self.current_score})")
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
        except ImportError:
            print("Matplotlib not available for visualization")