"""
Metrics and losses.
"""

from collections import Counter

import torch as tc


def cross_entropy_loss(logits, labels):
    return tc.nn.CrossEntropyLoss()(input=logits, target=labels)


def top_k_err(logits, labels, k):
    topk_preds = tc.topk(logits, k=k, dim=-1).indices
    matches = tc.eq(topk_preds, labels.unsqueeze(-1)).float().sum(dim=-1)
    acc_at_k = matches.mean(dim=0)
    return 1. - acc_at_k


def compute_losses_and_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    top1_err = top_k_err(logits, labels, k=1)
    top5_err = top_k_err(logits, labels, k=5)
    return {
        "loss": loss,
        "top1_err": top1_err,
        "top5_err": top5_err
    }


def global_mean(metric, world_size):
    # for logging purposes only!
    global_metric = metric.clone().float().detach()
    tc.distributed.all_reduce(global_metric, op=tc.distributed.ReduceOp.SUM)
    return global_metric.item() / world_size


def global_means(metrics, world_size):
    # for logging purposes only!
    return Counter({k: global_mean(v, world_size) for k, v in metrics.items()})
