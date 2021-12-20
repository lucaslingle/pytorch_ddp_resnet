"""Script."""

import argparse
import os

import torch as tc

from resnet.architectures.residual_block import ResNet
from resnet.algos.training import training_loop
from resnet.algos.evaluation import evaluation_loop

from resnet.utils.data_util import get_dataloaders
from resnet.utils.checkpoint_util import maybe_load_checkpoint


def create_argparser():
    parser = argparse.ArgumentParser(
        description="A Pytorch implementation of Deep Residual Networks, " +
                    "using Torch Distributed Data Parallel.")

    ### distributed
    parser.add_argument("--mode", choices=['train', 'eval'], default='train')
    parser.add_argument("--backend", choices=['gloo', 'mpi', 'nccl'], default='gloo')
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--master_addr", type=str, default='localhost')
    parser.add_argument("--master_port", type=int, default=12345)

    ### training hparams
    parser.add_argument("--dataset", choices=['cifar10', 'cifar100', 'imagenet'])
    # TODO(lucaslingle): add data augmentation choices argument, and implement code for it.
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--global_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dampening", type=float, default=0.0)
    parser.add_argument("--nesterov", choices=[0,1], default=0)

    ### persistence
    parser.add_argument("--checkpoint_dir", type=str, default='models_dir')
    parser.add_argument("--run_name", type=str, default='default_hparams')
    parser.add_argument("--data_dir", type=str, default="data_dir")
    return parser


def setup(rank, config):
    os.environ['MASTER_ADDR'] = config.get('master_addr')
    os.environ['MASTER_PORT'] = str(config.get('master_port'))
    tc.distributed.init_process_group(
        backend=config.get('backend'),
        world_size=config.get('world_size'),
        rank=rank)

    dl_train, dl_test = get_dataloaders(
        data_dir=config.get('data_dir'),
        batch_size=config.get('batch_size'))

    device = f'cuda:{rank}' if tc.cuda.is_available() else 'cpu'
    classifier = tc.nn.parallel.DistributedDataParallel(
        ResNet().to(device))
    optimizer = tc.optim.Adam(classifier.parameters(), lr=config.get('lr'))
    a = maybe_load_checkpoint(
        checkpoint_dir=config.get('checkpoint_dir'),
        run_name=config.get('run_name'),
        kind_name='classifier',
        checkpointable=classifier,
        steps=None)
    b = maybe_load_checkpoint(
        checkpoint_dir=config.get('checkpoint_dir'),
        run_name=config.get('run_name'),
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
        rank=rank,
        world_size=config.get('world_size'),
        device=learning_system.get('device'),
        classifier=learning_system.get('classifier'),
        optimizer=learning_system.get('optimizer'),
        dataloader=learning_system.get('dl_train'),
        global_step=learning_system.get('global_step'),
        max_steps=config.get('max_steps'),
        checkpoint_dir=config.get('checkpoint_dir'),
        run_name=config.get('run_name'))
    cleanup()


def evaluate(rank, config):
    learning_system = setup(rank, config)
    if rank == 0:
        metrics = evaluation_loop(
            device=learning_system.get('device'),
            classifier=learning_system.get('classifier'),
            dataloader=learning_system.get('dl_test'))
        print(f"Test loss: {metrics['loss']}... Test accuracy: {metrics['acc']}")
    cleanup()


if __name__ == '__main__':
    args = create_argparser().parse_args()
    config = vars(args)
    if args.mode == 'train':
        tc.multiprocessing.spawn(
            train,
            args=(config,),
            nprocs=args.world_size,
            join=True)
    else:
        tc.multiprocessing.spawn(
            evaluate,
            args=(config,),
            nprocs=args.world_size,
            join=True)
