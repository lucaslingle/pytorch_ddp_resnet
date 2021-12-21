"""Script."""

import argparse
import os

import torch as tc

from resnet.architectures.resnet import ResNet
from resnet.algos.training import training_loop
from resnet.algos.evaluation import evaluation_loop

from resnet.utils.config_util import ConfigParser
from resnet.utils.data_util import get_dataloaders
from resnet.utils.lr_util import get_scheduler
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


def get_config(args):
    base_path = os.path.join(args.models_dir, args.run_name)
    config_path = os.path.join(base_path, 'config.yaml')
    checkpoint_dir = os.path.join(base_path, 'checkpoints')
    log_dir = os.path.join(base_path, 'tensorboard_logs')

    config = ConfigParser(
        defaults={
            'checkpoint_dir': checkpoint_dir,
            'log_dir': log_dir,
            'data_dir': args.data_dir
        }
    )
    config.read(config_path, verbose=True)
    return config


def maybe_load_checkpoints(checkpoint_dir, classifier, optimizer, scheduler):
    a = maybe_load_checkpoint(
        checkpoint_dir=checkpoint_dir,
        kind_name='classifier',
        checkpointable=classifier,
        steps=None)
    b = maybe_load_checkpoint(
        checkpoint_dir=checkpoint_dir,
        kind_name='optimizer',
        checkpointable=optimizer,
        steps=None)
    c = maybe_load_checkpoint(
        checkpoint_dir=checkpoint_dir,
        kind_name='scheduler',
        checkpointable=scheduler,
        steps=None)
    if a != b or (c is not None and b != c):
        msg = "Latest checkpoint steps not aligned."
        raise RuntimeError(msg)
    return a


def setup(rank, config):
    os.environ['MASTER_ADDR'] = config.get('master_addr')
    os.environ['MASTER_PORT'] = config.get('master_port')
    tc.distributed.init_process_group(
        backend=config.get('backend'),
        world_size=config.get('world_size'),
        rank=rank)

    dl_train, dl_test = get_dataloaders(
        data_dir=config.get('data_dir'),
        dataset_name=config.get('dataset_name'),
        data_aug=config.get('data_aug'),
        batch_size=config.get('batch_size') // config.get('world_size'))

    device = f"cuda:{rank}" if tc.cuda.is_available() else "cpu"
    classifier = tc.nn.parallel.DistributedDataParallel(
        ResNet(
            architecture_spec=config.get('architecture_spec'),
            preact=config.get('preact'),
            use_proj=config.get('use_proj'),
            dropout_prob=config.get('dropout_prob')
        ).to(device)
    )
    optimizer = tc.optim.SGD(
        classifier.parameters(),
        lr=config.get('lr'),
        momentum=config.get('momentum'),
        dampening=config.get('dampening'),
        nesterov=config.get('nesterov'),
        weight_decay=config.get('weight_decay'))
    scheduler = get_scheduler(
        scheduler_cls_name=config.get('scheduler_cls_name'),
        optimizer=optimizer,
        scheduler_args=config.get('scheduler_args'))

    global_step = maybe_load_checkpoints(
        checkpoint_dir=config.get('checkpoint_dir'),
        classifier=classifier,
        optimizer=optimizer,
        scheduler=scheduler)

    return {
        "device": device,
        "dl_train": dl_train,
        "dl_test": dl_test,
        "classifier": classifier,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "global_step": a
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
            args=(config,),
            nprocs=config.get('world_size'),
            join=True)
