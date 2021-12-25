"""
Evaluation loop.
"""

from collections import Counter

import torch as tc

from resnet.algos.metrics import compute_losses_and_metrics, global_means


@tc.no_grad()
def evaluation_loop(
        world_size,
        classifier,
        dl_test,
        device,
        **kwargs
):
    classifier.eval()
    summed_metrics = Counter()
    num_batch = 0
    for x, y in dl_test:
        x, y = x.to(device), y.to(device)
        logits = classifier(x)
        metrics = compute_losses_and_metrics(logits=logits, labels=y)

        for name in metrics:
            summed_metrics[name] += metrics.get(name)
        num_batch += 1

    metrics = {k: v / num_batch for k,v in summed_metrics.items()}
    return global_means(metrics, world_size)

