"""
Metrics and losses.
"""

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
    top1 = top_k_err(logits, labels, k=1)
    top5 = top_k_err(logits, labels, k=5)
    return {
        "loss": loss,
        "top1": top1,
        "top5": top5
    }
