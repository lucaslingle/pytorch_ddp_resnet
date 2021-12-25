"""
Evaluation loop.
"""

from typing import Dict, Any
from collections import Counter

import torch as tc

from resnet.algos.metrics import compute_losses_and_metrics, global_means
from resnet.utils.types_util import Module, Dataloader


@tc.no_grad()
def evaluation_loop(
        world_size: int,
        classifier: Module,
        dl_test: Dataloader,
        device: str,
        **kwargs: Dict[str, Any]
) -> Dict[str, float]:
    """
    Evaluates classifier on the validation/test set.

    :param world_size: World size.
    :param classifier: Classifier.
    :param dl_test: Test/val dataloader.
    :param device: Device.
    :param kwargs: Keyword arguments
    :return: Dictionary of global metric floats, keyed by name.
    """
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
    return {k: v.item() for k,v in global_means(metrics, world_size)}

