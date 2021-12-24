"""
Evaluation loop.
"""

from collections import Counter

import torch as tc

from resnet.algos.metrics import compute_losses_and_metrics


@tc.no_grad()
def evaluation_loop(
        classifier,
        dl_test,
        device,
        **kwargs
):
    classifier.eval()
    summed_metrics = Counter()
    num_batch = 0
    for i, (x, y) in enumerate(dl_test):
        x, y = x.to(device), y.to(device)
        logits = classifier(x)
        metrics = compute_losses_and_metrics(logits=logits, labels=y)

        for name in metrics:
            summed_metrics[name] += metrics.get(name)
        num_batch += 1

    return {
        k: v / num_batch for k,v in summed_metrics.items()
    }
