"""
Data util.
"""

from typing import Tuple, Dict, Union, Any
import importlib
import inspect
import os
from collections import OrderedDict

import torch as tc
import torchvision as tv
from filelock import FileLock

from resnet.utils.checkpoint_util import maybe_load_checkpoint, save_checkpoint
from resnet.utils.types_util import Dataloader


def _get_transform(transform_cls_name, **kwargs):
    module = importlib.import_module('resnet.utils.transform_util')
    cls = getattr(module, transform_cls_name)
    return cls(**kwargs)


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
    return dataset_cls(**kwargs)


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

    # critical section for multiple processes, handled via a file-based lock.
    with FileLock(lock_fp):
        transforms_train = OrderedDict()
        prev_transform = None

        for transform_cls_name, transform_kwargs in data_aug.items():
            # we update dataset with new transforms because there could be
            # multiple FittedTransforms whose fitting process depends on inputs.
            dataset_train_ = _get_dataset(
                dataset_cls_name, root=data_dir, train=True, download=True,
                transform=tv.transforms.Compose(transforms_train.values()))

            if prev_transform is None:
                data_shape = dataset_train_.data[0].shape
            else:
                # since transforms are lazily applied, case above only works to
                # retrieve raw data tensor shape, not processed!
                data_shape = prev_transform.output_shape

            transform = _get_transform(
                transform_cls_name=transform_cls_name,
                data_shape=data_shape,
                **transform_kwargs)

            # first process to enter critical section runs for-loop to completion,
            # checkpointing all fittable transforms along the way.
            if isinstance(transform, FittableTransform):
                step = maybe_load_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    kind_name=transform_cls_name.lower(),
                    checkpointable=transform,
                    steps=None)
                if step == 0:
                    transform.fit(dataset=dataset_train_)
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        kind_name=transform_cls_name.lower(),
                        checkpointable=transform,
                        steps=1)

            transforms_train[transform_cls_name] = transform
            prev_transform = transform

        # todo(lucaslingle): add separate composed transform for validation set.
        #     could simply be whitening transform on cifar,
        #     but could also be five crop on imagenet for instance.
        dataset_train = _get_dataset(
            dataset_cls_name, root=data_dir, train=True, download=True,
            transform=tv.transforms.Compose(transforms_train.values()))

        dataset_test = _get_dataset(
            dataset_cls_name, root=data_dir, train=False, download=True,
            transform=tv.transforms.Compose(transforms_train.values()))

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
