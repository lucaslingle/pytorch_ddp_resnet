"""
Data util.
"""

from typing import Tuple, Dict, Union, Any
import os
import importlib

import numpy as np
import scipy as sp
import torch as tc
import torchvision as tv
from filelock import FileLock
import PIL

from resnet.utils.checkpoint_util import maybe_load_checkpoint, save_checkpoint
from resnet.utils.types_util import Module, Dataloader


def _get_dataset(dataset_cls_name, **kwargs):
    module = importlib.import_module('torchvision.datasets')
    dataset = getattr(module, dataset_cls_name)
    return dataset(**kwargs)


class _ZeroMeanWhiteningTransform(tc.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_fitted', tc.tensor(False))
        self._rgb_mean = tc.nn.Parameter(
            tc.zeros(size=(3,), dtype=tc.float32),
            requires_grad=False)

    def fit(self, dataset: tc.utils.data.Dataset) -> None:
        num_items = 0
        rgb_mean = np.zeros(shape=(3,), dtype=np.float32)
        for x, y in dataset:
            x = np.array(x).astype(np.float32)
            rgb_mean += np.mean(x, axis=(0, 1))
            num_items += 1
        rgb_mean /= num_items
        self._rgb_mean.copy_(tc.tensor(rgb_mean).float())
        self.register_buffer('_fitted', tc.tensor(True))

    def forward(self, x):
        assert self._fitted
        x = tc.tensor(np.array(x).astype(np.float32)).float()
        shift = self._rgb_mean.reshape(1, 1, 3)
        whitened = (x - shift)
        print(whitened.shape)
        return PIL.Image.fromarray(whitened.detach().numpy().astype(np.uint8))


class _StandardizeWhiteningTransform(tc.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_fitted', tc.tensor(False))
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
            rgb_mean += np.mean(x, axis=(0,1))
            num_items += 1
        rgb_mean /= num_items

        shift = rgb_mean.reshape([1, 1, 3])
        for x, y in dataset:
            x = np.array(x).astype(np.float32)
            rgb_var += np.mean(np.square(x-shift), axis=(0,1))
        rgb_var /= num_items
        rgb_stddev = np.sqrt(rgb_var)

        self._rgb_mean.copy_(tc.tensor(rgb_mean).float())
        self._rgb_stddev.copy_(tc.tensor(rgb_stddev).float())
        self.register_buffer('_fitted', tc.tensor(True))

    def forward(self, x):
        assert self._fitted
        x = tc.tensor(np.array(x).astype(np.float32)).float()
        shift = self._rgb_mean.reshape(1, 1, 3)
        scale = self._rgb_stddev.reshape(1, 1, 3)
        whitened = (x - shift) / scale
        return PIL.Image.fromarray(whitened.detach().numpy().astype(np.uint8))


class _ZCAWhiteningTransform(tc.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_fitted', tc.tensor(False))
        self._zca_matrix = tc.nn.Parameter(
            tc.zeros(size=(3,3), dtype=tc.float32), requires_grad=False)

    def fit(self, dataset: tc.utils.data.Dataset) -> None:
        num_items = 0
        rgb_mean = np.zeros(shape=(3,), dtype=np.float32)
        rgb_cov = np.zeros(shape=(3,3), dtype=np.float32)
        for x, y in dataset:
            x = np.array(x).astype(np.float32)
            rgb_mean += np.mean(x, axis=(0,1))
        rgb_mean /= num_items

        shift = rgb_mean.reshape([1, 1, 3])
        for x, y in dataset:
            x = np.array(x).astype(np.float32)
            vec = np.mean((x - shift), axis=(0,1))
            rgb_cov += np.outer(vec, vec)
        rgb_cov /= num_items
        rgb_cov_sqrt = sp.linalg.sqrtm(rgb_cov)

        self._zca_matrix.copy_(tc.tensor(rgb_cov_sqrt).float())
        self.register_buffer('_fitted', tc.tensor(True))

    def forward(self, x):
        assert self._fitted
        x = tc.tensor(np.array(x).astype(np.float32)).float()
        whitened = tc.matmul(x, self._zca_matrix.T)
        return PIL.Image.fromarray(whitened.detach().numpy().astype(np.uint8))


class _IdentityTransform(tc.nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, dataset: tc.utils.data.Dataset) -> None:
        return None

    def forward(self, x):
        return x


def _get_whitening_transform(whitening: str) -> Module:
    """
    :param whitening: One of 'zeromean', 'standardize', 'zca', 'none'.
    :return: Whitening transform function.
    """
    if whitening == 'zeromean':
        return _ZeroMeanWhiteningTransform()
    if whitening == 'standardized':
        return _StandardizeWhiteningTransform()
    if whitening == 'zca':
        return _ZCAWhiteningTransform()
    if whitening == 'none':
        return _IdentityTransform()


def _get_flip_transform(flip: str) -> Module:
    """
    :param flip: One of 'horizontal', 'none'.
    :return: Flip transform.
    """
    if flip == 'horizontal':
        return tv.transforms.RandomHorizontalFlip(p=0.5)
    if flip == 'none':
        return _IdentityTransform()


def _get_padding_transform(pad_width: int, pad_type: str) -> Module:
    """
    :param pad_width: Number of pixels to pad each size by.
    :param pad_type: One of 'zero', 'mirror', 'none'.
    :return: Padding transform.
    """
    if pad_type == 'zero':
        return tv.transforms.Pad(
            padding=(pad_width, pad_width), fill=0, padding_mode='constant')
    if pad_type == 'mirror':
        return tv.transforms.Pad(
            padding=(pad_width, pad_width), fill=0, padding_mode='reflect')
    if pad_type == 'none':
        return _IdentityTransform()


def _get_crop_transform(crop_size: int) -> Module:
    """
    :param crop_size: Pixel size to crop to.
    :return: Crop transform.
    """
    return tv.transforms.RandomCrop(size=(crop_size, crop_size))


def get_dataloaders(
        rank: int,
        data_dir: str,
        dataset_cls_name: str,
        data_aug: Dict[str, Union[int, str]],
        checkpoint_dir: str,
        local_batch_size: int,
        num_shards: int,
        **kwargs: Dict[str, Any]
) -> Tuple[Dataloader, Dataloader]:
    """
    Downloads data, builds preprocessing pipeline, shards it,
        shuffles it, and batches it via dataloaders.

    :param rank: Process rank.
    :param data_dir: Data directory to save downloaded datasets to.
    :param dataset_cls_name: Dataset class name in torchvision.datasets.
    :param data_aug: Data augmentation dictionary.
    :param checkpoint_dir: Checkpoint directory to save fitted whitening transforms.
    :param local_batch_size: Batch size per process.
    :param num_shards: Number of shards for the data.
    :param kwargs: Keyword arguments.
    :return: Tuple of train and test dataloaders,
        with data sharded appropriately per-process.
    """
    os.makedirs(data_dir, exist_ok=True)
    lock_fp = os.path.join(data_dir, f"{dataset_cls_name}.lock")
    with FileLock(lock_fp):
        dataset_train_ = _get_dataset(
            dataset_cls_name, root=data_dir, train=True, download=True,
            transform=None)

        # get whitening transform and fit it to the data.
        whitening_transform = _get_whitening_transform(
            whitening=data_aug.get('whitening'))
        step = maybe_load_checkpoint(
            checkpoint_dir=checkpoint_dir,
            kind_name='whitening',
            checkpointable=whitening_transform,
            steps=None)
        if step == 0:
            # file will stay locked til one process finishes doing everything.
            # that process will fit the whitening transform,
            # and the other processes will load it later,
            # in which case the step will not be zero anymore.
            whitening_transform.fit(dataset=dataset_train_)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                kind_name='whitening',
                checkpointable=whitening_transform,
                steps=1)

        # get the other transforms.
        flip_transform = _get_flip_transform(
            flip=data_aug.get('flip'))
        padding_transform = _get_padding_transform(
            pad_width=data_aug.get('pad_width'),
            pad_type=data_aug.get('pad_type'))
        crop_transform = _get_crop_transform(
            crop_size=data_aug.get('crop_size'))

        # torchvision's native tensor transform scales everything down by 255,
        # which leaves zero padding unaffected. moreover, the whitening ops
        # mathematically commute with this scaling.
        tensor_transform = tv.transforms.ToTensor()
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
        sampler_train = tc.utils.data.DistributedSampler(
            dataset=dataset_train,
            num_replicas=num_shards,
            rank=rank,
            shuffle=True,
            seed=0,
            drop_last=False)
        sampler_test = tc.utils.data.DistributedSampler(
            dataset=dataset_test,
            num_replicas=num_shards,
            rank=rank,
            shuffle=True,
            seed=0,
            drop_last=False)
    else:
        sampler_train = None
        sampler_test = None

    dataloader_train = tc.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=local_batch_size,
        shuffle=(sampler_train is None),
        sampler=sampler_train)
    dataloader_test = tc.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=local_batch_size,
        shuffle=(sampler_test is None),
        sampler=sampler_test)

    return dataloader_train, dataloader_test
