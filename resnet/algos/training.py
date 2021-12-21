"""
Training loop.
"""

import torch as tc
from torch.utils.tensorboard import SummaryWriter

from resnet.utils.checkpoint_util import save_checkpoint


def training_loop(
        rank,
        world_size,
        classifier,
        optimizer,
        dl_train,
        dl_test,
        device,
        global_step,
        max_steps,
        checkpoint_dir,
        log_dir
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
            x = x.to(device)
            y = y.to(device)

            logits = classifier(x)
            loss = tc.nn.CrossEntropyLoss()(input=logits, target=y)
            acc = tc.eq(logits.argmax(dim=-1), y).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_loss = global_mean(loss)
            global_acc = global_mean(acc)

            if rank == 0:
                writer.add_scalar('train/loss', global_loss.item(), global_step)
                writer.add_scalar('train/acc', global_acc.item(), global_step)

                if global_step % 100 == 0:
                    print(f"global step: {global_step}... loss: {global_loss.item()}")
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        kind_name='classifier',
                        checkpointable=classifier,
                        rank=rank,
                        steps=global_step+1)
                    save_checkpoint(
                        models_dir=checkpoint_dir,
                        kind_name='optimizer',
                        checkpointable=optimizer,
                        rank=rank,
                        steps=global_step+1)

            global_step += 1
            if done():
                break
