"""
算法模块
"""

from .eco import ECOAlgorithm, create_eco_config
from .s2v import S2VAlgorithm, create_s2v_config

__all__ = [
    'ECOAlgorithm',
    'create_eco_config',
    'S2VAlgorithm', 
    'create_s2v_config'
]