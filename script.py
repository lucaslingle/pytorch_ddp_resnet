"""Script."""

import argparse
import os
import configparser
import sys

import torch as tc

from resnet.architectures.resnet import ResNet
from resnet.algos.training import training_loop
from resnet.algos.evaluation import evaluation_loop

from resnet.utils.data_util import get_dataloaders
from resnet.utils.checkpoint_util import maybe_load_checkpoint


def create_argparser():
    parser = argparse.ArgumentParser(
        description="A Pytorch implementation of Deep Residual Networks, " +
                    "using Torch Distributed Data Parallel.")

    parser.add_argument("--mode", choices=['train', 'eval'], default='train')
    parser.add_argument("--models_dir", type=str, default='models_dir')
    parser.add_argument("--run_name", type=str, default='default_hparams')
    parser.add_argument("--data_dir", type=str, default="data_dir")
    return parser


def persistence_spec(models_dir, run_name):
    config_path = os.path.join(models_dir, run_name, 'config.ini')
    checkpoint_dir = os.path.join(models_dir, run_name, 'checkpoints')
    log_dir = os.path.join(models_dir, run_name, 'tensorboard_logs')
    return {
        "config_path": config_path,
        "checkpoint_dir": checkpoint_dir,
        "log_dir": log_dir
    }


def setup(rank, config):
    os.environ['MASTER_ADDR'] = config.get('master_addr')
    os.environ['MASTER_PORT'] = str(config.get('master_port'))
    tc.distributed.init_process_group(
        backend=config.get('backend'),
        world_size=int(config.get('world_size')),
        rank=rank)

    dl_train, dl_test = get_dataloaders(
        data_dir=config.get('data_dir'),
        batch_size=int(config.get('batch_size')))  # todo: make sure this is local batch size in configs

    device = f'cuda:{rank}' if tc.cuda.is_available() else 'cpu'
    classifier = tc.nn.parallel.DistributedDataParallel(
        ResNet().to(device))
    optimizer = tc.optim.SGD(
        classifier.parameters(),
        lr=float(config.get('lr')),
        momentum=float(config.get('momentum')),
        dampening=float(config.get('dampening')),
        nesterov=config.getbool('nesterov'),
        weight_decay=float(config.get('weight_decay')))
    a = maybe_load_checkpoint(
        checkpoint_dir=config.get('checkpoint_dir'),
        kind_name='classifier',
        checkpointable=classifier,
        steps=None)
    b = maybe_load_checkpoint(
        checkpoint_dir=config.get('checkpoint_dir'),
        kind_name='optimizer',
        checkpointable=optimizer,
        steps=None)
    if a != b:
        msg = "Latest classifier and optimizer checkpoint steps not aligned."
        raise RuntimeError(msg)

    return {
        "device": device,
        "dl_train": dl_train,
        "dl_test": dl_test,
        "classifier": classifier,
        "optimizer": optimizer,
        "global_step": a
    }


def cleanup():
    tc.distributed.destroy_process_group()


def train(rank, config):
    learning_system = setup(rank, config)
    training_loop(
        rank=rank, **config, **learning_system)
    cleanup()


def evaluate(rank, config):
    learning_system = setup(rank, config)
    if rank == 0:
        metrics = evaluation_loop(**config, **learning_system)
        print(f"Test loss: {metrics['loss']}... Test accuracy: {metrics['acc']}")
    cleanup()


if __name__ == '__main__':
    args = create_argparser().parse_args()
    persist_spec = persistence_spec(args.models_dir, args.run_name)

    config = configparser.ConfigParser()
    config.read(persist_spec.get('config_path'))
    config.read_dict({'DEFAULT': persist_spec})
    config = config['DEFAULT']

    for key in config:
        print(f"{key}: {config[key]}")

    if args.mode == 'train':
        tc.multiprocessing.spawn(
            train,
            args=(config,),
            nprocs=int(config.get('world_size')),
            join=True)
    else:
        tc.multiprocessing.spawn(
            evaluate,
            args=(config,),
            nprocs=int(config.get('world_size')),
            join=True)
