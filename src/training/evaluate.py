import torch

from training.metrics import (
    reduce_sum
)


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            targets
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        preds = (
            torch.sigmoid(outputs)
            >= 0.5
        ).float()

        correct += (
            preds == targets
        ).sum().item()

        total += targets.size(0)

    running_loss = reduce_sum(
        running_loss,
        device
    )

    correct = reduce_sum(
        correct,
        device
    )

    total = reduce_sum(
        total,
        device
    )

    return (
        running_loss / total,
        correct / total
    )