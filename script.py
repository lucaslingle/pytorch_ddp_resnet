"""Script."""

import argparse
import os

import torch as tc

from resnet.architectures.resnet import ResNet
from resnet.algos.training import training_loop
from resnet.algos.evaluation import evaluation_loop

from resnet.utils.config_util import ConfigParser
from resnet.utils.data_util import get_datasets, get_samplers, get_dataloaders
from resnet.utils.optim_util import get_optimizer, get_scheduler
from resnet.utils.checkpoint_util import (
    get_checkpoint_strategy, maybe_load_checkpoints
)


def create_argparser():
    parser = argparse.ArgumentParser(
        description="A Pytorch implementation of Deep Residual Networks, " +
                    "using Torch Distributed Data Parallel.")

    parser.add_argument("--mode", choices=['train', 'eval'], default='train')
    parser.add_argument("--models_dir", type=str, default='models_dir')
    parser.add_argument("--run_name", type=str, default='wrn-28-10-dropout_cifar10')
    parser.add_argument("--data_dir", type=str, default='data_dir')
    return parser


def get_config(args):
    base_path = os.path.join(args.models_dir, args.run_name)
    config_path = os.path.join(base_path, 'config.yaml')
    checkpoint_dir = os.path.join(base_path, 'checkpoints')
    log_dir = os.path.join(base_path, 'tensorboard_logs')

    config = ConfigParser(
        defaults={
            'mode': args.mode,
            'data_dir': args.data_dir,
            'checkpoint_dir': checkpoint_dir,
            'log_dir': log_dir
        }
    )
    config.read(config_path, verbose=True)
    return config


def setup(rank, config):
    os.environ['MASTER_ADDR'] = config.get('master_addr')
    os.environ['MASTER_PORT'] = config.get('master_port')
    tc.distributed.init_process_group(
        backend=config.get('backend'),
        world_size=config.get('world_size'),
        rank=rank)

    datasets = get_datasets(**config)
    samplers = get_samplers(rank, **config, **datasets)
    dataloaders = get_dataloaders(**config, **datasets, **samplers)

    device = f"cuda:{rank}" if tc.cuda.is_available() else "cpu"
    scaler = tc.cuda.amp.GradScaler() if tc.cuda.is_available() else None
    classifier = tc.nn.parallel.DistributedDataParallel(
        ResNet(
            architecture_spec=config.get('architecture_spec'),
            preact=config.get('preact'),
            use_proj=config.get('use_proj'),
            dropout_prob=config.get('dropout_prob')
        ).to(device)
    )
    optimizer = get_optimizer(
        model=classifier,
        optimizer_cls_name=config.get('optimizer_cls_name'),
        optimizer_args=config.get('optimizer_args'))
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_cls_name=config.get('scheduler_cls_name'),
        scheduler_args=config.get('scheduler_args'))
    checkpoint_strategy = get_checkpoint_strategy(
        checkpoint_strategy_cls_name=config.get('checkpoint_strategy_cls_name'),
        checkpoint_strategy_args=config.get('checkpoint_strategy_args'))

    global_step = maybe_load_checkpoints(
        checkpoint_dir=config.get('checkpoint_dir'),
        checkpointables={
            'scaler': scaler,
            'classifier': classifier,
            'optimizer': optimizer,
            'scheduler': scheduler
        },
        map_location=device,
        steps=None)

    return {
        "device": device,
        "sampler_train": samplers.get('sampler_train'),
        "sampler_test": samplers.get('sampler_test'),
        "dl_train": dataloaders.get('dl_train'),
        "dl_test": dataloaders.get('dl_test'),
        "classifier": classifier,
        "optimizer": optimizer,
        "scaler": scaler,
        "scheduler": scheduler,
        "checkpoint_strategy": checkpoint_strategy,
        "global_step": global_step
    }


def cleanup():
    tc.distributed.destroy_process_group()


def train(rank, config):
    learning_system = setup(rank, config)
    training_loop(rank, **config, **learning_system)
    cleanup()


def evaluate(rank, config):
    learning_system = setup(rank, config)
    metrics = evaluation_loop(**config, **learning_system)
    if rank == 0:
        print(f"Test metrics: {metrics}")
    cleanup()


if __name__ == '__main__':
    args = create_argparser().parse_args()
    config = get_config(args)
    tc.multiprocessing.spawn(
        train if config.get('mode') == 'train' else evaluate,
        args=(config,),
        nprocs=config.get('world_size'),
        join=True)
