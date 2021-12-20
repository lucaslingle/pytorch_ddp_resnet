"""
Residual networks.
"""

import torch as tc


class ResidualBlock(tc.nn.Module):
    def __init__(
            self,
            channels: int,
            downsample: bool,
            preact: str,
            use_proj: bool
    ):
        """
        :param channels: Number of input channels. 
        :param downsample: Downsample by factor of two?
        :param preact: A string specifying whether to use preactivation ordering.
        :param use_proj: Use projection on skip connection when downsampling?
        """
        super().__init__()
        self._in_channels = channels
        self._out_channels = channels if not downsample else 2 * channels
        self._downsample = downsample
        self._preact = preact
        self._use_proj = use_proj

        self._conv1 = tc.nn.Conv2d(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            kernel_size=(3,3),
            stride=(1,1) if not self._downsample else (2,2)
        )
        self._conv2 = tc.nn.Conv2d(
            in_channels=self._out_channels,
            out_channels=self._out_channels,
            kernel_size=(3,3),
            stride=(1,1) if not self._downsample else (2,2)
        )
        if self._downsample and self._use_proj:
            self._proj = tc.nn.Conv2d(
                in_channels=self._in_channels,
                out_channels=self._out_channels,
                kernel_size=(1,1),
                stride=(1,1)
            )

        self._norm1 = tc.nn.BatchNorm2d(
            num_features=self._in_channels if self._preact else self._out_channels)
        self._norm2 = tc.nn.BatchNorm2d(
             num_features=self._out_channels)

        self._act1 = tc.nn.ReLU()
        self._act2 = tc.nn.ReLU()

    def forward(self, x):
        i = x
        if self._preact:
            x = self._norm1(x)
            x = self._act1(x)
            x = self._conv1(x)
            x = self._norm2(x)
            x = self._act2(x)
            x = self._conv2(x)
        else:
            x = self._conv1(x)
            x = self._norm1(x)
            x = self._act1(x)
            x = self._conv2(x)
            x = self._norm2(x)

        if self._downsample:
            if self._use_proj:
                i = self._proj(i)
            else:
                i = tc.nn.functional.pad(i, (0,0,0,self._channels))

        h = i + x
        if not self._preact:
            h = self._act2(h)
        return h
