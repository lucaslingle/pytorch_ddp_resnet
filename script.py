"""Script."""

import argparse
import os

import torch as tc

from resnet.architectures.resnet import ResNet
from resnet.algos.training import training_loop
from resnet.algos.evaluation import evaluation_loop

from resnet.utils.config_util import ConfigParser
from resnet.utils.data_util import get_dataloaders
from resnet.utils.optim_util import get_optimizer, get_scheduler
from resnet.utils.checkpoint_util import maybe_load_checkpoints


def create_argparser():
    parser = argparse.ArgumentParser(
        description="A Pytorch implementation of Deep Residual Networks, " +
                    "using Torch Distributed Data Parallel.")

    parser.add_argument("--mode", choices=['train', 'eval'], default='train')
    parser.add_argument("--models_dir", type=str, default='models_dir')
    parser.add_argument("--run_name", type=str, default='default_hparams')
    parser.add_argument("--data_dir", type=str, default="data_dir")
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


def get_shard_spec(mode, batch_size, world_size):
    if mode == 'train':
        num_shards = world_size
    else:
        num_shards = 1
    local_batch_size = batch_size // num_shards
    return {
        "num_shards": num_shards,
        "local_batch_size": local_batch_size
    }


def setup(rank, config):
    os.environ['MASTER_ADDR'] = config.get('master_addr')
    os.environ['MASTER_PORT'] = config.get('master_port')
    tc.distributed.init_process_group(
        backend=config.get('backend'),
        world_size=config.get('world_size'),
        rank=rank)

    shard_spec = get_shard_spec(
        mode=config.get('mode'),
        batch_size=config.get('batch_size'),
        world_size=config.get('world_size'))
    dl_train, dl_test = get_dataloaders(
        rank=rank,
        data_dir=config.get('data_dir'),
        dataset_cls_name=config.get('dataset_cls_name'),
        data_aug=config.get('data_aug'),
        checkpoint_dir=config.get('checkpoint_dir'),
        local_batch_size=shard_spec.get('local_batch_size'),
        num_shards=shard_spec.get('num_shards'))

    device = f"cuda:{rank}" if tc.cuda.is_available() else "cpu"
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

    global_step = maybe_load_checkpoints(
        checkpoint_dir=config.get('checkpoint_dir'),
        checkpointables={
            'classifier': classifier,
            'optimizer': optimizer,
            'scheduler': scheduler
        },
        steps=None)

    return {
        "device": device,
        "dl_train": dl_train,
        "dl_test": dl_test,
        "classifier": classifier,
        "optimizer": optimizer,
        "scheduler": scheduler,
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
    if rank == 0:
        metrics = evaluation_loop(**config, **learning_system)
        print(f"Test loss: {metrics['loss']}... Test accuracy: {metrics['acc']}")
    cleanup()


if __name__ == '__main__':
    args = create_argparser().parse_args()
    config = get_config(args)

    if args.mode == 'train':
        tc.multiprocessing.spawn(
            train,
            args=(config,),
            nprocs=config.get('world_size'),
            join=True)
    else:
        tc.multiprocessing.spawn(
            evaluate,
            args=config.get('world_size'),
            nprocs=1,
            join=True)
