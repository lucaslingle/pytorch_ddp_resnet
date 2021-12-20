"""
Residual networks.
"""

from typing import Tuple
import re

import torch as tc

from residual_block import ResidualBlock


def extract_ints(text: str, num: int) -> Tuple[int]:
    pattern = r"(\w+)" + r",".join([r"([0-9]+)" for _ in range(num)])
    m = re.match(pattern, text)
    ints = tuple(map(lambda x: int(x), m.groups()[1:]))
    return ints


class ResNet(tc.nn.Module):
    def __init__(
            self,
            architecture_spec: str,
            preact: bool,
            use_proj: bool,
            dropout_prob: float
    ):
        """
        A residual network.

        :param architecture_spec: A string containing a space-separated list of
            architectural components chosen from
                {"cK,S,P,I,O", "mpK,S,P", "apK,S,P", "rD", "bD", "n", "a", "fI,O}.
            c stands for convolution,
            mp for max pooling,
            ap for average pooling,
            r for residual block,
            b for bottleneck residual block,
            n for batch normalization,
            a for activation,
            f for fully-connected.
            K,S,P stand for kernel size, stride, padding and should be integers.
            D stands for depth: the number of applications of the given block type.
                If two stacks of residual blocks are listed adjacently,
                the first block in the second stack always performs downsampling.
            I,O stand for input, output dim.
        :param preact: Use preactivation ordering?
        :param use_proj: Use projection on skip connection when downsampling?
        :param dropout_prob: Dropout probability.
        """
        super().__init__()
        self._architecture_spec = architecture_spec
        self._preact = preact
        self._use_proj = use_proj
        self._dropout_prob = dropout_prob

        self._architecture = self._parse_spec(architecture_spec)

    def _make_conv(self, k, s, p, i, o):
        return tc.nn.Conv2d(
            in_channels=i,
            out_channels=o,
            kernel_size=(k, k),
            stride=(s, s),
            padding=(p, p))

    def _make_avgpool(self, k, s, p):
        return tc.nn.AvgPoool2d(
            kernel_size=(k, k),
            stride=(s, s),
            padding=(p, p))

    def _make_maxpool(self, k, s, p):
        return tc.nn.MaxPoool2d(
            kernel_size=(k, k),
            stride=(s, s),
            padding=(p, p))

    def _make_res_stack(self, d, i, o, l):
        return tc.nn.Sequential(*[
            ResidualBlock(
                channels=i if ell == 0 else o,
                downsample=d if ell == 0 else False,
                preact=self._preact,
                use_proj=self._use_proj,
                dropout_prob=self._dropout_prob)
            for ell in range(l)
        ])

    def _make_bottleneck_res_stack(self, d, i, o, l):
        raise NotImplementedError

    def _make_norm(self, i):
        return tc.nn.BatchNorm2d(num_features=i)

    def _make_act(self):
        return tc.nn.ReLU()

    def _make_fc(self, i, o):
        return tc.nn.Sequential(
            tc.nn.Flatten(),
            tc.nn.Linear(i, o))

    def _parse_spec(self, spec: str) -> tc.nn.Sequential:
        ms = list()
        channels = None
        for n, component in enumerate(spec.split()):
            if component.startswith('c'):
                kspio = extract_ints(component, 5)
                m = self._make_conv(*kspio)
                channels = kspio[-1]
            elif component.startswith('mp'):
                m = self._make_maxpool(*extract_ints(component, 3))
            elif component.startswith('ap'):
                m = self._make_avgpool(*extract_ints(component, 3))
            elif component.startswith('r'):
                d = spec.split()[n-1].startswith('r')
                i = channels
                o = channels if not d else 2 * channels
                l = extract_ints(component, 1)
                m = self._make_res_stack(d, i, o, l)
                channels = o
            elif component.startswith('b'):
                d = spec.split()[n-1].startswith('b')
                i = channels
                o = channels if not d else 2 * channels
                l = extract_ints(component, 1)
                m = self._make_bottleneck_res_stack(d, i, o, l)
                channels = o
            elif component.startswith('n'):
                i = channels
                m = self._make_norm(i)
            elif component.startswith('a'):
                m = self._make_act()
            elif component.startswith('f'):
                m = self._make_fc(*extract_ints(component, 2))
            else:
                raise ValueError("Unknown component in architecture spec.")
            ms.append(m)
        return tc.nn.Sequential(*ms)

    def forward(self, x):
        return self._architecture(x)
