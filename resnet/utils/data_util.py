"""
Data util.
"""

from typing import Tuple, Dict, Union
import os
import importlib

import numpy as np
import torch as tc
import torchvision as tv
from filelock import FileLock

from resnet.utils.types_util import Module


def _get_dataset(dataset_cls_name, **kwargs):
    module = importlib.import_module('torchvision.datasets')
    dataset = getattr(module, dataset_cls_name)
    return dataset(**kwargs)


def _format_to_reduction_indices(format: str) -> Tuple[int, int]:
    return {
        'CHW': (-2, -1),
        'HWC': (-3, -2)
    }[format]


class ZeroMeanWhiteningTransform(tc.nn.Module):
    def __init__(self, format):
        assert format in ['CHW', 'HWC']
        super().__init__()
        self._reduction_indices = _format_to_reduction_indices(format)
        self._fitted = False
        self._rgb_mean = tc.nn.Parameter(
            tc.zeros(size=(3,), dtype=tc.float32),
            requires_grad=False)

    def fit(self, dataset: tc.utils.data.Dataset) -> None:
        num_items = 0
        rgb_mean = np.zeros(shape=(3,), dtype=np.float32)
        for x, y in dataset:
            x = np.array(x).astype(np.float32)
            rgb_mean += np.mean(x, axis=self._reduction_indices)
            num_items += 1
        self._rgb_mean.copy_(tc.tensor(rgb_mean / num_items).float())
        self._fitted = True

    def forward(self, x):
        assert self._fitted
        shift = self._rgb_mean[:, None, None]
        return x - shift


class StandardizeWhiteningTransform(tc.nn.Module):
    def __init__(self, format):
        assert format in ['CHW', 'HWC']
        super().__init__()
        self._reduction_indices = _format_to_reduction_indices(format)
        self._fitted = False
        self._rgb_mean = tc.nn.Parameter(
            tc.zeros(size=(3,), dtype=tc.float32), requires_grad=False)
        self._rgb_stddev = tc.nn.Parameter(
            tc.ones(size=(3,), dtype=tc.float32), requires_grad=False)

    def fit(self, dataset: tc.utils.data.Dataset) -> None:
        num_items = 0
        rgb_mean = np.zeros(shape=(3,), dtype=np.float32)
        rgb_var = np.zeros(shape=(3,), dtype=np.float32)
        for x, y in dataset:
            x = np.array(x).astype(np.float32)
            rgb_mean += np.mean(x, axis=self._reduction_indices)
            num_items += 1
        rgb_mean /= num_items

        for x, y in dataset:
            x = np.array(x).astype(np.float32)
            rgb_var += np.mean(
                np.square(x-rgb_mean), axis=self._reduction_indices)
        rgb_var /= num_items
        rgb_stddev = np.sqrt(rgb_var)

        self._rgb_mean.copy_(tc.tensor(rgb_mean).float())
        self._rgb_stddev.copy_(tc.tensor(rgb_stddev).float())
        self._fitted = True

    def forward(self, x):
        assert self._fitted
        shift = self._rgb_mean[:, None, None]
        scale = self._rgb_stddev[:, None, None]
        return (x - shift) / scale


class ZCAWhiteningTransform(tc.nn.Module):
    def __init__(self, format):
        assert format in ['CHW', 'HWC']
        super().__init__()
        self._reduction_indices = _format_to_reduction_indices(format)
        self._fitted = False
        self._matrix = tc.nn.Parameter(
            tc.zeros(size=(3,3), dtype=tc.float32), requires_grad=False)

    def fit(self, dataset: tc.utils.data.Dataset) -> None:
        raise NotImplementedError

    def forward(self, x):
        assert self._fitted
        raise NotImplementedError


class IdentityTransform(tc.nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, dataset: tc.utils.data.Dataset) -> None:
        return None

    def forward(self, x):
        return x


def get_whitening_transform(
        whitening: str, format: str
) -> Module:
    """
    :param whitening: one of 'zeromean', 'standardize', 'zca', 'none'.
    :param format: one of 'CHW', 'HWC'
    :return: Whitening transform function.
    """
    if whitening == 'zeromean':
        return ZeroMeanWhiteningTransform(format=format)
    if whitening == 'standardized':
        return StandardizeWhiteningTransform(format=format)
    if whitening == 'zca':
        return ZCAWhiteningTransform(format=format)
    if whitening == 'none':
        return IdentityTransform()


def get_flip_transform(flip: str) -> Module:
    """
    :param flip: one of 'horizontal', 'none'
    :return: flip transform.
    """
    if flip == 'horizontal':
        return tv.transforms.RandomHorizontalFlip(p=0.5)
    if flip == 'none':
        return IdentityTransform()


def get_padding_transform(pad_width: int, pad_type: str) -> Module:
    """
    :param pad_width: number of pixels to pad each size by.
    :param pad_type: one of 'zero', 'mirror', 'none'.
    :return: padding transform.
    """
    if pad_type == 'zero':
        return tv.transforms.Pad(
            padding=(pad_width, pad_width), fill=0, padding_mode='constant')
    if pad_type == 'mirror':
        return tv.transforms.Pad(
            padding=(pad_width, pad_width), fill=0, padding_mode='reflect')
    if pad_type == 'none':
        return IdentityTransform()


def get_crop_transform(crop_size: int) -> Module:
    """
    :param crop_size: pixel size to crop to.
    :return: crop transform.
    """
    return tv.transforms.RandomCrop(size=(crop_size, crop_size))


def get_dataloaders(
        rank: int,
        data_dir: str,
        dataset_cls_name: str,
        data_aug: Dict[str, Union[int, str]],
        local_batch_size: int,
        num_shards: int
):
    os.makedirs(data_dir, exist_ok=True)
    lock_fp = os.path.join(data_dir, f"{dataset_cls_name}.lock")
    with FileLock(lock_fp):
        dataset_train_ = _get_dataset(
            dataset_cls_name, root=data_dir, train=True, download=True,
            transform=None)

        whitening_transform = get_whitening_transform(
            whitening=data_aug.get('whitening'))
        if True:
            # todo(lucaslingle):
            #    make a checkpoint check condition for whitening module.
            whitening_transform.fit(dataset_train=dataset_train_)
        else:
            # load checkpoint
            raise NotImplementedError

        flip_transform = get_flip_transform(
            flip=data_aug.get('flip'))
        padding_transform = get_padding_transform(
            pad_width=data_aug.get('pad_width'),
            pad_type=data_aug.get('pad_type'))
        crop_transform = get_crop_transform(
            crop_size=data_aug.get('crop_size'))
        tensor_transform = tv.transforms.ToTensor()

        # torchvision's native tensor transform scales everything down by 255,
        # which leaves zero padding unaffected. moreover, the whitening ops
        # mathematically commute with this scaling.
        composed_transform = tv.transforms.Compose([
            whitening_transform,
            flip_transform,
            padding_transform,
            crop_transform,
            tensor_transform
        ])

        dataset_train = _get_dataset(
            dataset_cls_name, root=data_dir, train=True, download=True,
            transform=composed_transform)

        dataset_test = _get_dataset(
            dataset_cls_name, root=data_dir, train=False, download=True,
            transform=composed_transform)

    if num_shards > 1:
        dataset_train = tc.utils.data.DistributedSampler(
            dataset=dataset_train,
            num_replicas=num_shards,
            rank=rank,
            shuffle=True,
            seed=0,
            drop_last=False)
        dataset_test = tc.utils.data.DistributedSampler(
            dataset=dataset_test,
            num_replicas=num_shards,
            rank=rank,
            shuffle=True,
            seed=0,
            drop_last=False)

    dataloader_train = tc.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=local_batch_size,
        shuffle=True)
    dataloader_test = tc.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=local_batch_size,
        shuffle=False)

    return dataloader_train, dataloader_test
