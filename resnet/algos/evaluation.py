"""
Evaluation loop.
"""

import torch as tc


@tc.no_grad()
def evaluation_loop(
        classifier,
        dl_test,
        device,
        **kwargs
):
    classifier.eval()
    summed_loss = 0.
    summed_acc = 0.
    num_batch = 0
    for x, y in dl_test:
        x = x.to(device)
        y = y.to(device)

        logits = classifier(x)
        loss = tc.nn.CrossEntropyLoss()(input=logits, target=y)
        acc = tc.eq(logits.argmax(dim=-1), y).float().mean()

        loss = loss.item()
        acc = acc.item()
        summed_loss += loss
        summed_acc += acc
        num_batch += 1

    mean_loss = summed_loss / num_batch
    mean_acc = summed_acc / num_batch

    return {
        "loss": mean_loss,
        "acc": mean_acc
    }