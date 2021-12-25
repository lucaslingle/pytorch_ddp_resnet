"""
Data util.
"""

from typing import Dict, Union, Any, Optional
import importlib
import inspect
import contextlib
import os
from collections import OrderedDict

import torch as tc
import torchvision as tv
from filelock import FileLock

from resnet.utils.checkpoint_util import maybe_load_checkpoint, save_checkpoint
from resnet.utils.transform_util import Transform, FittableTransform
from resnet.utils.types_util import Dataset, Sampler, Dataloader


def _get_transform_cls(transform_cls_name):
    module = importlib.import_module('resnet.utils.transform_util')
    cls = getattr(module, transform_cls_name)
    return cls


def _get_dataset(dataset_cls_name, **kwargs):
    module = importlib.import_module('torchvision.datasets')
    dataset_cls = getattr(module, dataset_cls_name)
    if 'split' in inspect.signature(dataset_cls).parameters:
        kwargs['split'] = 'train' if kwargs['train'] else 'val'
        del kwargs['train']
    if dataset_cls_name == 'ImageNet':
        del kwargs['download']
        # todo(lucaslingle):
        #    check if it's not downloaded, and then download imagenet here
    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f):
            dataset = dataset_cls(**kwargs)
    return dataset


def _get_initial_data_shape(data_dir, dataset_cls_name):
    dataset_train_ = _get_dataset(
        dataset_cls_name, root=data_dir, train=True, download=True,
        transform=None)
    return dataset_train_.data[0].shape


def _get_transforms(
        data_dir: str,
        dataset_cls_name: str,
        data_aug: Dict[str, Dict[str, Union[str, int, float]]],
        checkpoint_dir: str,
        is_train: bool,
        reusable_transforms: OrderedDict[str, Transform],
) -> OrderedDict[str, Transform]:
    """
    Creates an ordered dictionary of Transforms.

    :param data_dir: Data directory to save downloaded datasets to.
    :param dataset_cls_name: Dataset class name in torchvision.datasets.
    :param data_aug: Data augmentation/processing spec.
    :param checkpoint_dir: Checkpoint directory to save fitted transforms.
    :param is_train: Boolean indicating train or test/validation set.
    :param reusable_transforms: OrderedDict of reusable transforms.
    :return: OrderedDict of Transforms.
    """
    transforms = OrderedDict()
    data_shape = _get_initial_data_shape(data_dir, dataset_cls_name)
    for transform_cls_name, transform_kwargs in data_aug.items():
        # we update dataset with new transforms because there could be
        # multiple FittedTransforms whose fitting process depends on inputs.
        dataset = _get_dataset(
            dataset_cls_name, root=data_dir, train=True, download=True,
            transform=tv.transforms.Compose(transforms.values()))

        transform_cls = _get_transform_cls(transform_cls_name)
        transform = transform_cls(data_shape, **transform_kwargs)
        if is_train:
            if isinstance(transform, FittableTransform):
                step = maybe_load_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    kind_name=transform_cls_name.lower(),
                    checkpointable=transform,
                    map_location='cpu',
                    steps=None)
                if step == 0:
                    transform.fit(dataset=dataset)
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        kind_name=transform_cls_name.lower(),
                        checkpointable=transform,
                        steps=1)
        else:
            if isinstance(transform, FittableTransform):
                if transform_cls_name not in reusable_transforms:
                    msg = "Fittable test transform not in reusable_transforms."
                    raise ValueError(msg)

                transform = reusable_transforms[transform_cls_name]
                if transform.data_shape != data_shape:
                    msg = "Input shape mismatch on reusable transform."
                    raise ValueError(msg)

        transforms[transform_cls_name] = transform
        data_shape = transform.output_shape
    return transforms


def get_datasets(
        data_dir: str,
        dataset_cls_name: str,
        data_aug_train: Dict[str, Dict[str, Union[str, int, float]]],
        data_aug_test: Dict[str, Dict[str, Union[str, int, float]]],
        checkpoint_dir: str,
        **kwargs: Dict[str, Any]
) -> Dict[str, Dataset]:
    """
    Downloads data and builds preprocessing pipeline.

    :param data_dir: Data directory to save downloaded datasets to.
    :param dataset_cls_name: Dataset class name in torchvision.datasets.
    :param data_aug_train: Training data augmentation/processing spec.
    :param data_aug_test: Test data augmentation/processing spec.
    :param checkpoint_dir: Checkpoint directory to save fitted transforms.
    :param kwargs: Keyword arguments.
    :return: Dictionary of train and test datasets.
    """
    os.makedirs(data_dir, exist_ok=True)
    lock_fp = os.path.join(data_dir, f"{dataset_cls_name}.lock")
    with FileLock(lock_fp):
        transforms_train = _get_transforms(
            data_dir=data_dir, dataset_cls_name=dataset_cls_name,
            data_aug=data_aug_train, checkpoint_dir=checkpoint_dir,
            is_train=True, reusable_transforms=OrderedDict())

        transforms_test = _get_transforms(
            data_dir=data_dir, dataset_cls_name=dataset_cls_name,
            data_aug=data_aug_test, checkpoint_dir=checkpoint_dir,
            is_train=False, reusable_transforms=transforms_train)

        dataset_train = _get_dataset(
            dataset_cls_name, root=data_dir, train=True, download=True,
            transform=tv.transforms.Compose(transforms_train.values()))

        dataset_test = _get_dataset(
            dataset_cls_name, root=data_dir, train=False, download=True,
            transform=tv.transforms.Compose(transforms_test.values()))

        return {
            'dataset_train': dataset_train,
            'dataset_test': dataset_test
        }


def get_samplers(
        rank: int,
        world_size: int,
        dataset_train: Dataset,
        dataset_test: Dataset,
        **kwargs: Dict[str, Any]
) -> Dict[str, Sampler]:
    """
    Instantiates distributed samplers.

    :param rank: Process rank.
    :param world_size: Number of processes.
    :param dataset_train: Training dataset,
    :param dataset_test: Validation/test dataset.
    :param kwargs: Keyword args.
    :return: Dictionary of optional train and test samplers.
    """
    sampler_train = tc.utils.data.DistributedSampler(
        dataset=dataset_train,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=0,
        drop_last=False)
    sampler_test = tc.utils.data.DistributedSampler(
        dataset=dataset_test,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=0,
        drop_last=False)

    return {
        'sampler_train': sampler_train,
        'sampler_test': sampler_test
    }


def get_dataloaders(
        dataset_train: Dataset,
        dataset_test: Dataset,
        sampler_train: Sampler,
        sampler_test: Sampler,
        batch_size: int,
        world_size: int,
        **kwargs: Dict[str, Any]
) -> Dict[str, Dataloader]:
    """
    Instantiates dataloaders.

    :param dataset_train: Training dataset,
    :param dataset_test: Validation/test dataset.
    :param sampler_train: This process' distributed sampler for training set.
    :param sampler_test: This process' distributed sampler for training set.
    :param world_size: Number of processes.
    :param batch_size: Global batch size.
    :param kwargs: Keyword args.
    :return: Dictionary of dataloaders for train and test data.
    """
    local_batch_size = batch_size // world_size

    dl_train = tc.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler_train)
    dl_test = tc.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler_test)

    return {
        'dl_train': dl_train,
        'dl_test': dl_test
    }
