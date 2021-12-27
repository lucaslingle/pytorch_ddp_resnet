"""
Training loop.
"""

from typing import Optional, Dict, Any, Union

import torch as tc
from torch.utils.tensorboard import SummaryWriter

from resnet.algos.metrics import compute_losses_and_metrics, global_means
from resnet.algos.evaluation import evaluation_loop
from resnet.utils.checkpoint_util import CheckpointStrategy, save_checkpoints
from resnet.utils.types_util import (
    Sampler, Dataloader, Module, Optimizer, Scaler, Scheduler
)


def requires_loss(scheduler: Scheduler) -> bool:
    return isinstance(scheduler, tc.optim.lr_scheduler.ReduceLROnPlateau)


def step_scheduler(scheduler: Scheduler, loss: Union[tc.Tensor, float]) -> None:
    if requires_loss(scheduler):
        scheduler.step(loss)
    else:
        scheduler.step()


def training_loop(
        rank: int,
        world_size: int,
        device: str,
        sampler_train: Sampler,
        sampler_test: Sampler,
        dl_train: Dataloader,
        dl_test: Dataloader,
        classifier: Module,
        optimizer: Optimizer,
        scaler: Optional[Scaler],
        scheduler: Optional[Scheduler],
        scheduler_step_unit: str,
        checkpoint_strategy: CheckpointStrategy,
        checkpoint_dir: str,
        global_step: int,
        max_steps: int,
        log_dir: str,
        **kwargs: Dict[str, Any]
) -> None:
    """
    Runs a training loop on a given process.

    :param rank: Process rank.
    :param world_size: World size.
    :param device: Device name.
    :param sampler_train: Training distributed sampler.
    :param sampler_test: Test/val distributed sampler.
    :param dl_train: Training dataloader.
    :param dl_test: Test/val dataloader.
    :param classifier: Classifier.
    :param optimizer: Optimizer.
    :param grad_scaler: Optional grad scaler for mixed-precision training.
    :param scheduler: Optional learning rate scheduler.
    :param scheduler_step_unit: One of 'batch', 'epoch', 'none'.
    :param checkpoint_strategy: CheckpointStrategy instance.
    :param checkpoint_dir: Checkpoint directory to save checkpoints to.
    :param global_step: Global step to start from.
    :param max_steps: Maximum number of steps.
    :param log_dir: Logging directory for Tensorboard.
    :param kwargs: Keyword args.
    :return:
    """

    if rank == 0:
        writer = SummaryWriter(log_dir)

    def done():
        return global_step >= max_steps

    while not done():
        # todo(lucaslingle): use something more reliable to estimate epoch,
        #  see warning at link https://pytorch.org/docs/stable/data.html
        epoch = (global_step // len(dl_train))
        sampler_train.set_epoch(epoch)

        classifier.train()
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            with tc.cuda.amp.autocast(enabled=scaler is not None):
                logits = classifier(x)
                metrics = compute_losses_and_metrics(logits=logits, labels=y)
                loss = metrics.get('loss')

            if scaler:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            global_metrics = global_means(metrics, world_size)
            global_loss = global_metrics.get('loss')

            if scheduler and scheduler_step_unit == 'batch':
                step_scheduler(scheduler, global_loss)

            if rank == 0:
                print(f"global step: {global_step}... loss: {global_loss}")
                for name in global_metrics:
                    writer.add_scalar(
                        tag=f"train/{name}",
                        scalar_value=global_metrics.get(name),
                        global_step=global_step)

                if checkpoint_strategy.is_eligible(
                        unit='batch', step=global_step, loss=global_loss):
                    save_checkpoints(
                        checkpoint_dir=checkpoint_dir,
                        checkpointables={
                            'classifier': classifier,
                            'optimizer': optimizer,
                            'scheduler': scheduler,
                            'scaler': scaler
                        },
                        steps=global_step+1)

            global_step += 1
            if done():
                break

        global_val_metrics = evaluation_loop(world_size, classifier, dl_test, device)
        global_val_loss = global_val_metrics.get('loss')

        if scheduler and scheduler_step_unit == 'epoch':
            step_scheduler(scheduler, global_val_loss)

        if rank == 0:
            print(f"epoch: {epoch}... validation loss: {global_val_loss}")
            for name in global_val_metrics:
                writer.add_scalar(
                    tag=f"val/{name}",
                    scalar_value=global_val_metrics.get(name),
                    global_step=epoch)

            if checkpoint_strategy.is_eligible(
                    unit='epoch', step=epoch, loss=global_val_loss):
                save_checkpoints(
                    checkpoint_dir=checkpoint_dir,
                    checkpointables={
                        'classifier': classifier,
                        'optimizer': optimizer,
                        'scheduler': scheduler,
                        'scaler': scaler
                    },
                    steps=global_step+1)
