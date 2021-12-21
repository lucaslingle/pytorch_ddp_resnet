"""
Training loop.
"""

import torch as tc
from torch.utils.tensorboard import SummaryWriter

from resnet.algos.metrics import compute_losses_and_metrics
from resnet.utils.checkpoint_util import save_checkpoint


def training_loop(
        rank,
        world_size,
        classifier,
        optimizer,
        scheduler,
        dl_train,
        dl_test,
        device,
        global_step,
        max_steps,
        checkpoint_dir,
        log_dir,
        **kwargs
):
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

            optimizer.zero_grad()
            metrics.get('loss').backward()
            optimizer.step()

            global_metrics = {k: global_mean(v) for k,v in metrics.items()}
            if rank == 0:
                for name in global_metrics:
                    writer.add_scalar(
                        tag=f"train/{name}",
                        scalar_value=global_metrics.get(name).item(),
                        global_step=global_step)

                if global_step % 100 == 0:
                    global_loss = global_metrics.get('loss').item()
                    print(f"global step: {global_step}... loss: {global_loss}")
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        kind_name='classifier',
                        checkpointable=classifier,
                        rank=rank,
                        steps=global_step+1)
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        kind_name='optimizer',
                        checkpointable=optimizer,
                        rank=rank,
                        steps=global_step+1)
                    if scheduler:
                        save_checkpoint(
                            checkpoint_dir=checkpoint_dir,
                            kind_name='scheduler',
                            checkpointable=scheduler,
                            rank=rank,
                            steps=global_step+1)

            global_step += 1
            if done():
                break
