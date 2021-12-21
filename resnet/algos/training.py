"""
Training loop.
"""

from typing import Optional, Dict, Any
import inspect

import torch as tc
from torch.utils.tensorboard import SummaryWriter

from resnet.algos.metrics import compute_losses_and_metrics
from resnet.algos.evaluation import evaluation_loop
from resnet.utils.checkpoint_util import save_checkpoints
from resnet.utils.types_util import Module, Optimizer, Scheduler, Dataloader


def step_scheduler(scheduler, loss):
    if hasattr(inspect.signature(scheduler.step), 'metrics'):
        scheduler.step(loss)
    else:
        scheduler.step()


def training_loop(
        rank: int,
        world_size: int,
        classifier: Module,
        optimizer: Optimizer,
        scheduler: Optional[Scheduler],
        scheduler_step_unit: str,
        dl_train: Dataloader,
        dl_test: Dataloader,
        device: str,
        global_step: int,
        max_steps: int,
        checkpoint_dir: str,
        log_dir: str,
        **kwargs: Dict[str, Any]
) -> None:
    """
    Runs a training loop on a given process.
    If rank is zero, logs metrics to tensorboard and saves checkpoints periodically.

    :param rank: Rank.
    :param world_size: World size.
    :param classifier: Classifier.
    :param optimizer: Optimizer.
    :param scheduler: Optional learning rate scheduler.
    :param scheduler_step_unit: One of 'batch', 'epoch', 'none'.
    :param dl_train: Training dataloader.
    :param dl_test: Test/val dataloader.
    :param device: Device name.
    :param global_step: Global step to start from.
    :param max_steps: Maximum number of steps.
    :param checkpoint_dir: Checkpoint directory to save checkpoints to.
    :param log_dir: Logging directory for Tensorboard.
    :param kwargs: Keyword args.
    :return:
    """

    if rank == 0:
        writer = SummaryWriter(log_dir)

    def global_mean(metric):  # for logging
        global_metric = metric.detach()
        tc.distributed.reduce(
            global_metric, dst=0, op=tc.distributed.ReduceOp.SUM)
        return global_metric / world_size

    def done():
        return global_step >= max_steps

    while not done():
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            logits = classifier(x)
            metrics = compute_losses_and_metrics(logits=logits, labels=y)
            loss = metrics.get('loss')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler_step_unit == 'batch':
                step_scheduler(scheduler, loss)

            # todo(lucaslingle): sort out maybe scheduler step
            #    issues include step freq (batch vs epoch)
            #    and passed args (loss required for ReduceLROnPlateau, for example).

            global_metrics = {k: global_mean(v) for k,v in metrics.items()}
            if rank == 0:
                for name in global_metrics:
                    writer.add_scalar(
                        tag=f"train/{name}",
                        scalar_value=global_metrics.get(name).item(),
                        global_step=global_step)

                if global_step % 100 == 0:
                    # todo(lucaslingle): add support for sophisticated checkpointing strategies.
                    global_loss = global_metrics.get('loss').item()
                    print(f"global step: {global_step}... loss: {global_loss}")
                    save_checkpoints(
                        checkpoint_dir=checkpoint_dir,
                        checkpointables={
                            'classifier': classifier,
                            'optimizer': optimizer,
                            'scheduler': scheduler
                        },
                        rank=rank,
                        steps=global_step+1)

            global_step += 1
            if done():
                break

        if scheduler_step_unit == 'epoch':
            # todo(lucaslingle): sort out logic for communicating global loss on validation set, especially as it may be sharded over multiple processes.
            # when done, add comment explaining why you're doing what you're doing
            val_metrics = evaluation_loop(classifier, dl_test, device)
            val_loss = val_metrics.get('loss')
            val_loss_global = global_mean(val_loss)
            step_scheduler(scheduler, val_loss_global)
