"""
Residual networks.
"""

import torch as tc


class ResNet(tc.nn.Module):
    def __init__(self, dataset_name):
        super().__init__()