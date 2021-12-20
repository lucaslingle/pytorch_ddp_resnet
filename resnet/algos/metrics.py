"""
Metrics and losses.
"""

import torch as tc


def compute_losses_and_metrics(logits, labels):
    loss = tc.nn.CrossEntropyLoss()(input=logits, target=labels)
    acc = tc.eq(logits.argmax(dim=-1), labels).float().mean()
    return {
        "loss": loss,
        "acc": acc
    }
