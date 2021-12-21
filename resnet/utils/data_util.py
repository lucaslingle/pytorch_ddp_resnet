"""
Data util.
"""

import os

import torch as tc
import torchvision as tv
from filelock import FileLock


def get_dataloaders(data_dir, dataset_name, data_aug, batch_size):
    os.makedirs(data_dir, exist_ok=True)
    lock_fp = os.path.join(data_dir, f"{dataset_name}.lock")
    with FileLock(lock_fp):
        dataset_train = tv.datasets.FashionMNIST(
            root=data_dir, train=True, download=True,
            transform=tv.transforms.ToTensor())

        dataset_test = tv.datasets.FashionMNIST(
            root=data_dir, train=False, download=True,
            transform=tv.transforms.ToTensor())

    # TODO(lucaslingle):
    #    shard the data over multiple gpus, and add support for data aug
    if data_aug is not None:
        raise NotImplementedError("Gotta implement data aug later!")

    dataloader_train = tc.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = tc.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_test
