"""
Data util.
"""

from typing import Dict, Union, Any, Optional
import importlib
import inspect
import os
from collections import OrderedDict

import torch as tc
import torchvision as tv
from filelock import FileLock

from resnet.utils.checkpoint_util import maybe_load_checkpoint, save_checkpoint
from resnet.utils.transform_util import FittableTransform
from resnet.utils.types_util import Dataset, Sampler, Dataloader


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


def get_datasets(
        device: str,
        data_dir: str,
        dataset_cls_name: str,
        data_aug: Dict[str, Union[int, str]],
        checkpoint_dir: str,
        **kwargs: Dict[str, Any]
) -> Dict[str, Dataset]:
    """
    Downloads data and builds preprocessing pipeline.

    :param device: Device for fittable transforms.
    :param data_dir: Data directory to save downloaded datasets to.
    :param dataset_cls_name: Dataset class name in torchvision.datasets.
    :param data_aug: Data augmentation dictionary.
    :param checkpoint_dir: Checkpoint directory to save fitted whitening transforms.
    :param kwargs: Keyword arguments.
    :return: Dictionary of train and test datasets.
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
                **transform_kwargs
            ).to(device)

            # first process to enter critical section runs for-loop to completion,
            # checkpointing all fittable transforms along the way.
            if isinstance(transform, FittableTransform):
                step = maybe_load_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    kind_name=transform_cls_name.lower(),
                    checkpointable=transform,
                    map_location=device,
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

        return {
            'dataset_train': dataset_train,
            'dataset_test': dataset_test
        }


def get_samplers(
        rank: int,
        num_replicas: int,
        dataset_train: Dataset,
        dataset_test: Dataset,
        **kwargs: Dict[str, Any]
) -> Dict[str, Optional[Sampler]]:
    """
    Maybe instantiates distributed samplers.

    :param rank: Process rank.
    :param num_shards: Number of shards for the data.
    :param dataset_train: Training dataset,
    :param dataset_test: Validation/test dataset.
    :param kwargs: Keyword args.
    :return: Dictionary of optional train and test samplers.
    """
    if num_replicas > 1:
        sampler_train = tc.utils.data.DistributedSampler(
            dataset=dataset_train,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            seed=0,
            drop_last=False)
        sampler_test = tc.utils.data.DistributedSampler(
            dataset=dataset_test,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            seed=0,
            drop_last=False)
    else:
        sampler_train = None
        sampler_test = None

    return {
        'sampler_train': sampler_train,
        'sampler_test': sampler_test
    }


def get_dataloaders(
        dataset_train: Dataset,
        dataset_test: Dataset,
        sampler_train: Sampler,
        sampler_test: Sampler,
        local_batch_size: int,
        **kwargs: Dict[str, Any]
) -> Dict[str, Dataloader]:
    """
    Instantiates dataloaders.

    :param dataset_train: Training dataset,
    :param dataset_test: Validation/test dataset.
    :param sampler_train: This process' distributed sampler for training set.
    :param sampler_test: This process' distributed sampler for training set.
    :param local_batch_size: Batch size per process.
    :param kwargs: Keyword args.
    :return: Dictionary of dataloaders for train and test data.
    """
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
