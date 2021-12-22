"""
Data util.
"""

from typing import Dict, Union
import os
import importlib

import torch as tc
import torchvision as tv
from filelock import FileLock

from resnet.utils.types_util import Module


def get_dataset(dataset_cls_name, **kwargs):
    module = importlib.import_module('torchvision.datasets')
    dataset = getattr(module, dataset_cls_name)
    return dataset(**kwargs)


def get_whitening_transform(
        whitening: str,
        dataset_train: tc.utils.data.Dataset
) -> Module:
    """
    :param whitening: one of 'meanzero', 'standardized', 'zca', 'none'.
    :param dataset_train: training dataset without preprocessing.
    :return: Whitening transform function.
    """
    if whitening == 'meanzero':
        # compute mean over training data
        # return transform that subtracts it
        raise NotImplementedError
    if whitening == 'standardized':
        # compute mean and stddev over training data
        # return composed transform that normalizes data
        raise NotImplementedError
    if whitening == 'zca':
        # compute zca matrix over training data
        # return custom transform that applies it
        raise NotImplementedError
    if whitening == 'none':
        #return IdentityTransform() # todo: implement this class
        raise NotImplementedError


def get_flip_transform(flip: str) -> Module:
    """
    :param flip: one of 'horizontal', 'none'
    :return: flip transform.
    """
    if flip == 'horizontal':
        return tv.transforms.RandomHorizontalFlip(p=0.5)
    if flip == 'none':
        #return IdentityTransform() # todo: implement this class
        raise NotImplementedError


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
        # return IdentityTransform() # todo: implement this class
        raise NotImplementedError


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
        dataset_train_ = get_dataset(
            dataset_cls_name, root=data_dir, train=True, download=True,
            transform=None)

        whitening_transform = get_whitening_transform(
            whitening=data_aug.get('whitening'),
            dataset_train=dataset_train_)
        flip_transform = get_flip_transform(
            flip=data_aug.get('flip'))
        padding_transform = get_padding_transform(
            pad_width=data_aug.get('pad_width'),
            pad_type=data_aug.get('pad_type'))
        crop_transform = get_crop_transform(
            crop_size=data_aug.get('crop_size'))
        tensor_transform = tv.transforms.ToTensor()
        composed_transform = tv.transforms.Compose([
            whitening_transform,
            flip_transform,
            padding_transform,
            crop_transform,
            tensor_transform
        ])

        dataset_train = tv.datasets.FashionMNIST(
            root=data_dir, train=True, download=True,
            transform=composed_transform)

        dataset_test = tv.datasets.FashionMNIST(
            root=data_dir, train=False, download=True,
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
