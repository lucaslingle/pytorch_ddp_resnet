"""
Residual networks.
"""

import re

import torch as tc


class ResNet(tc.nn.Module):
    def __init__(
            self,
            architecture_spec,
            dropout_prob,
            num_classes
    ):
        """
        A residual network.

        :param architecture_spec: A string containing a space-separated list of
            architectural components chosen from {"cK,S,P", "pK,S,P", "rD", "bD", "n"}.
            c stands for convolution,
            p for average-pooling,
            r for residual block,
            b for bottleneck residual block,
            n for batch normalization.
            K,S,P stand for kernel size, stride, padding and should be integers.
            D stands for depth: the number of applications of the given block type.
            If two stacks of residual blocks are listed adjacently,
            the first block in the second stack always performs downsampling.
        :param dropout_prob: A dropout probability.
        """
        super().__init__()
        self._architecture_spec = architecture_spec
        self._dropout_prob = dropout_prob
        self._num_classes = num_classes

    def forward(self, x):
        raise NotImplementedError
