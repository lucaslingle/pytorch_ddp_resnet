"""
Residual networks.
"""

from typing import Tuple
import re

import torch as tc

from residual_block import ResidualBlock


def extract_ints(text, num):
    pattern = r"(\w+)" + r",".join([r"([0-9]+)" for _ in range(num)])
    m = re.match(pattern, text)
    ints = tuple(map(lambda x: int(x), m.groups()[1:]))
    return ints


class ResNet(tc.nn.Module):
    def __init__(
            self,
            img_shape: Tuple[int],
            architecture_spec: str,
            channels: int,
            preact: bool,
            use_proj: bool,
            dropout_prob: float
    ):
        """
        A residual network.

        :param architecture_spec: A string containing a space-separated list of
            architectural components chosen from
                {"cK,S,P", "mpK,S,P", "apK,S,P", "rD", "bD", "n", "a", "fI,O}.
            c stands for convolution,
            mp for max pooling,
            ap for average pooling.
            r for residual block,
            b for bottleneck residual block,
            n for batch normalization,
            a for activation.
            f for fully-connected.
            K,S,P stand for kernel size, stride, padding and should be integers.
            D stands for depth: the number of applications of the given block type.
                If two stacks of residual blocks are listed adjacently,
                the first block in the second stack always performs downsampling.
            I,O stand for input, output dim.
        :param: img_shape: Tuple containing image channels, height, width.
        :param channels: Number of output channels in first convolution.
        :param preact: Use preactivation ordering?
        :param use_proj: Use projection on skip connection when downsampling?
        :param dropout_prob: Dropout probability.
        """
        super().__init__()
        self._architecture_spec = architecture_spec
        self._img_shape = img_shape
        self._channels = channels
        self._preact = preact
        self._use_proj = use_proj
        self._dropout_prob = dropout_prob

        module_list = []
        channels = self._channels
        for i, component in enumerate(self._architecture_spec.split(" ")):
            if component.startswith('c'):
                k, s, p = extract_ints(component, 3)
                module = tc.nn.Conv2d(
                    in_channels=channels if i > 0 else self._img_shape[0],
                    out_channels=channels,
                    kernel_size=(k,k),
                    stride=(s,s),
                    padding=(p,p))
            elif component.startswith('mp'):
                k, s, p = extract_ints(component, 3)
                module = tc.nn.MaxPool2d(
                    kernel_size=(k, k),
                    stride=(s, s),
                    padding=(p, p))
            elif component.startswith('ap'):
                k, s, p = extract_ints(component, 3)
                module = tc.nn.AvgPool2d(
                    kernel_size=(k, k),
                    stride=(s, s),
                    padding=(p, p))
            elif component.startswith('r'):
                assert i != 0
                d = self._architecture_spec.split(" ")[i-1].startswith('r')
                module = ResidualBlock(
                    channels=channels,
                    downsample=d,
                    preact=self._preact,
                    use_proj=self._use_proj,
                    dropout_prob=self._dropout_prob)
            elif component.startswith('b'):
                assert i != 0
                raise NotImplementedError
            elif component.startswith('n'):
                module = tc.nn.BatchNorm2d(
                    num_features=channels)
            elif component.startswith('a'):
                module = tc.nn.ReLU()
            elif component.startswith('f'):
                i, o = extract_ints(component, 2)
                module = tc.nn.Linear(i, o)
            else:
                raise NotImplementedError
            module_list.append(module)

        self._architecture = tc.nn.ModuleList(module_list)

    def forward(self, x):
        return self._architecture(x)
