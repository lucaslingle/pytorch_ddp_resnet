"""
Residual blocks.
"""

import torch as tc


class ResidualBlock(tc.nn.Module):
    def __init__(
            self,
            channels: int,
            downsample: bool,
            preact: bool,
            use_proj: bool,
            dropout_prob: float
    ):
        """
        Basic residual block.

        :param channels: Number of input channels.
        :param downsample: Downsample by factor of two?
        :param preact: Use preactivation ordering?
        :param use_proj: Use projection on skip connection when downsampling?
        :param dropout_prob: Dropout probability.
        """
        super().__init__()
        self._in_channels = channels
        self._out_channels = channels if not downsample else 2 * channels
        self._downsample = downsample
        self._preact = preact
        self._use_proj = use_proj
        self._dropout_prob = dropout_prob

        self._conv1 = tc.nn.Conv2d(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            kernel_size=(3,3),
            stride=(1,1) if not self._downsample else (2,2),
            padding=(1,1),
            bias=False)
        self._conv2 = tc.nn.Conv2d(
            in_channels=self._out_channels,
            out_channels=self._out_channels,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
            bias=False)
        if self._downsample:
            self._pool = tc.nn.AvgPool2d(kernel_size=(1,1), stride=(2,2))
            if self._use_proj:
                self._proj = tc.nn.Conv2d(
                    in_channels=self._in_channels,
                    out_channels=self._out_channels,
                    kernel_size=(1,1),
                    stride=(1,1),
                    padding=(0,0),
                    bias=False)
        self._norm1 = tc.nn.BatchNorm2d(
            num_features=self._in_channels if self._preact else self._out_channels)
        self._norm2 = tc.nn.BatchNorm2d(
            num_features=self._out_channels)
        self._act1 = tc.nn.ReLU()
        self._act2 = tc.nn.ReLU()
        self._dropout1 = tc.nn.Dropout(p=self._dropout_prob, inplace=False)
        self._dropout2 = tc.nn.Dropout(p=self._dropout_prob, inplace=False)

    def forward(self, x):
        i = x
        if self._preact:
            x = self._norm1(x)
            x = self._act1(x)
            x = self._dropout1(x)
            x = self._conv1(x)

            x = self._norm2(x)
            x = self._act2(x)
            x = self._dropout2(x)
            x = self._conv2(x)
        else:
            x = self._dropout1(x)
            x = self._conv1(x)
            x = self._norm1(x)
            x = self._act1(x)

            x = self._dropout2(x)
            x = self._conv2(x)
            x = self._norm2(x)

        if self._downsample:
            i = self._pool(i)
            if self._use_proj:
                i = self._proj(i)
            else:
                i = tc.nn.functional.pad(i, (0,0,0,0,0,self._in_channels))

        h = i + x
        if not self._preact:
            h = self._act2(h)
        return h


class BottleneckResidualBlock(tc.nn.Module):
    def __init__(
            self,
            channels: int,
            downsample: bool,
            preact: bool,
            use_proj: bool,
            dropout_prob: float
    ):
        """
        Bottleneck residual block.

        :param channels: Number of input channels.
        :param downsample: Downsample by factor of two?
        :param preact: Use preactivation ordering?
        :param use_proj: Use projection on skip connection when downsampling?
        :param dropout_prob: Dropout probability.
        """
        super().__init__()
        self._in_channels = channels
        self._bottleneck_channels = channels // 4 if not downsample else channels // 2
        self._out_channels = channels if not downsample else 2 * channels
        self._downsample = downsample
        self._preact = preact
        self._use_proj = use_proj
        self._dropout_prob = dropout_prob

        self._conv1 = tc.nn.Conv2d(
            in_channels=self._in_channels,
            out_channels=self._bottleneck_channels,
            kernel_size=(1,1),
            stride=(1,1),
            padding=(0,0),
            bias=False)
        self._conv2 = tc.nn.Conv2d(
            in_channels=self._bottleneck_channels,
            out_channels=self._bottleneck_channels,
            kernel_size=(3,3),
            stride=(1,1) if not self._downsample else (2,2),
            padding=(1,1),
            bias=False)
        self._conv3 = tc.nn.Conv2d(
            in_channels=self._bottleneck_channels,
            out_channels=self._out_channels,
            kernel_size=(1,1),
            stride=(1,1),
            padding=(0,0),
            bias=False)
        if self._downsample:
            self._pool = tc.nn.AvgPool2d(kernel_size=(1,1), stride=(2,2))
            if self._use_proj:
                self._proj = tc.nn.Conv2d(
                    in_channels=self._in_channels,
                    out_channels=self._out_channels,
                    kernel_size=(1,1),
                    stride=(1,1),
                    padding=(0,0),
                    bias=False)
        self._norm1 = tc.nn.BatchNorm2d(
            num_features=self._in_channels if self._preact else self._bottleneck_channels)
        self._norm2 = tc.nn.BatchNorm2d(
            num_features=self._bottleneck_channels)
        self._norm3 = tc.nn.BatchNorm2d(
            num_features=self._bottleneck_channels if self._preact else self._out_channels)
        self._act1 = tc.nn.ReLU()
        self._act2 = tc.nn.ReLU()
        self._act3 = tc.nn.ReLU()
        self._dropout1 = tc.nn.Dropout(p=self._dropout_prob, inplace=False)
        self._dropout2 = tc.nn.Dropout(p=self._dropout_prob, inplace=False)
        self._dropout3 = tc.nn.Dropout(p=self._dropout_prob, inplace=False)

    def forward(self, x):
        i = x
        if self._preact:
            x = self._norm1(x)
            x = self._act1(x)
            x = self._dropout1(x)
            x = self._conv1(x)

            x = self._norm2(x)
            x = self._act2(x)
            x = self._dropout2(x)
            x = self._conv2(x)

            x = self._norm3(x)
            x = self._act3(x)
            x = self._dropout3(x)
            x = self._conv3(x)
        else:
            x = self._dropout1(x)
            x = self._conv1(x)
            x = self._norm1(x)
            x = self._act1(x)

            x = self._dropout2(x)
            x = self._conv2(x)
            x = self._norm2(x)
            x = self._act2(x)

            x = self._dropout3(x)
            x = self._conv3(x)
            x = self._norm3(x)

        if self._downsample:
            i = self._pool(i)
            if self._use_proj:
                i = self._proj(i)
            else:
                i = tc.nn.functional.pad(i, (0,0,0,0,0,self._in_channels))

        h = i + x
        if not self._preact:
            h = self._act3(h)
        return h
