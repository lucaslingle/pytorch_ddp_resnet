"""
Transform util.
"""

from typing import List
import abc
import math
import PIL

import torch as tc
import torchvision as tv

from resnet.utils.types_util import Dataset


class Transform(tc.nn.Module, abc.ABC):
    def __init__(self, data_shape):
        super().__init__()
        self._data_shape = data_shape

    @property
    def output_shape(self) -> List[int]:
        return list(self._data_shape)


class FittableTransform(Transform, abc.ABC):
    @abc.abstractmethod
    def fit(self, dataset: Dataset) -> None:
        raise NotImplementedError


class ZeroMeanWhiteningTransform(FittableTransform):
    def __init__(self, data_shape):
        super().__init__(data_shape)
        self._image_mean = tc.nn.Parameter(
            tc.zeros(size=data_shape, dtype=tc.float32),
            requires_grad=False)
        self.register_buffer('_fitted', tc.tensor(False))

    def fit(self, dataset: Dataset) -> None:
        mean = tc.zeros(size=self._data_shape, dtype=tc.float32)

        item_count = 1
        for x, y in dataset:
            mean *= (item_count-1) / item_count
            mean += x / item_count
            item_count += 1

        self._image_mean.copy_(mean)
        self.register_buffer('_fitted', tc.tensor(True))

    def forward(self, x: tc.Tensor) -> tc.Tensor:
        assert self._fitted
        whitened = x - self._image_mean
        return whitened


class StandardizeWhiteningTransform(FittableTransform):
    def __init__(self, data_shape):
        super().__init__(data_shape)
        self._image_mean = tc.nn.Parameter(
            tc.zeros(size=data_shape, dtype=tc.float32), requires_grad=False)
        self._image_stddev = tc.nn.Parameter(
            tc.ones(size=data_shape, dtype=tc.float32), requires_grad=False)
        self.register_buffer('_fitted', tc.tensor(False))

    def fit(self, dataset: Dataset) -> None:
        mean = tc.zeros(size=self._data_shape, dtype=tc.float32)
        var = tc.zeros(size=self._data_shape, dtype=tc.float32)

        item_count = 1
        for x, y in dataset:
            mean *= (item_count - 1) / item_count
            mean += x / item_count
            item_count += 1

        item_count = 1
        for x, y in dataset:
            var *= (item_count - 1) / item_count
            var += tc.square(x-mean) / item_count
            item_count += 1
        stddev = tc.sqrt(var)

        self._image_mean.copy_(mean)
        self._image_stddev.copy_(stddev)
        self.register_buffer('_fitted', tc.tensor(True))

    def forward(self, x: tc.Tensor) -> tc.Tensor:
        assert self._fitted
        whitened = (x - self._image_mean) / self._image_stddev
        return whitened


class ZCAWhiteningTransform(FittableTransform):
    def __init__(self, data_shape):
        super().__init__(data_shape)
        self._data_dim = math.prod(data_shape)
        self._zca_matrix = tc.nn.Parameter(
            tc.zeros(size=(self._data_dim, self._data_dim), dtype=tc.float32),
            requires_grad=False)
        self.register_buffer('_fitted', tc.tensor(False))

    @staticmethod
    def sqrtm(matrix, eps=1e-2):
        u, s, v = tc.svd(matrix)
        return tc.matmul(tc.matmul(u, tc.diag(tc.rsqrt(s + eps))), u.T)

    def fit(self, dataset: Dataset) -> None:
        mean = tc.zeros(size=(self._data_dim,), dtype=tc.float32)
        cov = tc.zeros(size=(self._data_dim, self._data_dim), dtype=tc.float32)

        item_count = 1
        for x, y in dataset:
            x = x.reshape(-1)
            mean *= (item_count - 1) / item_count
            mean += x / item_count
            item_count += 1

        item_count = 1
        for x, y in dataset:
            x = x.reshape(-1)
            vec = (x - mean)
            cov *= (item_count - 1) / item_count
            cov += tc.outer(vec, vec) / item_count
            item_count += 1
        zca_matrix = self.sqrtm(cov)

        self._zca_matrix.copy_(zca_matrix)
        self.register_buffer('_fitted', tc.tensor(True))

    def forward(self, x: tc.Tensor) -> tc.Tensor:
        assert self._fitted
        flat_white = tc.matmul(self._zca_matrix, x.reshape(-1))
        whitened = flat_white.reshape(self._data_shape)
        return whitened


class FlipTransform(Transform):
    def __init__(self, data_shape, p):
        super().__init__(data_shape)
        self._p = p

    def forward(self, x: tc.Tensor) -> tc.Tensor:
        probs = tc.tensor([1-self._p, self._p]).float()
        flip = tc.distributions.Categorical(probs=probs).sample().bool().item()
        if flip:
            return tc.flip(x, dims=(2,))
        return x


class PaddingTransform(Transform):
    def __init__(self, data_shape, pad_size, pad_type):
        assert pad_type in ['zero', 'mirror']
        super().__init__(data_shape)
        self._pad_size = pad_size
        self._pad_type = pad_type

    @property
    def output_shape(self) -> List[int]:
        c, h, w = self._data_shape
        h_new, w_new = map(lambda x: x + 2 * self._pad_size, [h, w])
        return [c, h_new, w_new]

    def forward(self, x: tc.Tensor) -> tc.Tensor:
        pad = tuple(self._pad_size for _ in range(4))
        if self._pad_type == 'mirror':
            return tc.nn.functional.pad(x, pad=pad, mode='reflect')
        elif self._pad_type == 'zero':
            return tc.nn.functional.pad(x, pad=pad, mode='constant', value=0.)


class RandomCropTransform(Transform):
    def __init__(self, data_shape, crop_size):
        super().__init__(data_shape)
        self._crop_size = crop_size

    @property
    def output_shape(self) -> List[int]:
        c, h, w = self._data_shape
        return [c, self._crop_size, self._crop_size]

    def forward(self, x: tc.Tensor) -> tc.Tensor:
        t_index_max = self._data_shape[1]-self._crop_size
        l_index_max = self._data_shape[2]-self._crop_size
        t_idx = tc.randint(low=0, high=t_index_max+1, size=(1,)).item()
        l_idx = tc.randint(low=0, high=l_index_max+1, size=(1,)).item()
        return x[:, t_idx:t_idx+self._crop_size, l_idx:l_idx+self._crop_size]


class ToTensorTransform(Transform):
    def __init__(self, data_shape):
        super().__init__(data_shape)
        self._transform = tv.transforms.ToTensor()

    @property
    def output_shape(self):
        h, w, c = self._data_shape
        return [c, h, w]

    def forward(self, x: PIL.Image) -> tc.Tensor:
        return self._transform(x)
