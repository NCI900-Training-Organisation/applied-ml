import torch
from training.metrics import (
    reduce_sum
)


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):
    """
    DDP (multi-GPU) training epoch.

    Key idea vs single-GPU:
    - Each GPU runs this SAME function independently.
    - Each GPU sees a DIFFERENT subset of the dataset.
    - Gradients are automatically synchronized by DDP during backward().
    - Metrics (loss/accuracy) must be manually reduced across GPUs.
    """

    model.train()

    # These are LOCAL metrics (per GPU process)
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:

        # Each process moves its batch to its own GPU
        # (assigned via LOCAL_RANK in setup_ddp)
        images = images.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )

        # Clear gradients for THIS GPU/process only
        optimizer.zero_grad()

        # Forward pass is independent per GPU
        outputs = model(images)

        # Loss computed locally on this GPU's mini-batch
        loss = criterion(
            outputs,
            targets
        )

        # BACKWARD PASS (IMPORTANT DDP BEHAVIOR):
        # --------------------------------------
        # In DDP, loss.backward() does TWO things:
        # 1. Computes gradients locally on each GPU
        # 2. Automatically all-reduces gradients across GPUs
        #
        # This ensures every GPU ends up with identical gradients
        # before optimizer.step().
        loss.backward()

        # Apply synchronized gradients
        optimizer.step()

        # ----------------------------------------------------
        # LOCAL METRIC TRACKING (per GPU only at this stage)
        # ----------------------------------------------------

        # Weighted loss accumulation for correct averaging later
        running_loss += (
            loss.item()
            * images.size(0)
        )

        # Prediction computation (still local)
        preds = (
            torch.sigmoid(outputs)
            >= 0.5
        ).float()

        # Count correct predictions on THIS GPU only
        correct += (
            preds == targets
        ).sum().item()

        # Count samples seen by THIS GPU only (data shard)
        total += targets.size(0)

    # --------------------------------------------------------
    # GLOBAL SYNCHRONIZATION STEP (CRITICAL DIFFERENCE)
    # --------------------------------------------------------
    #
    # At this point:
    # - GPU0 has partial metrics
    # - GPU1 has partial metrics
    # - GPU2 has partial metrics
    #
    # Each metric is ONLY valid locally until reduced.
    #
    # reduce_sum() performs:
    #   dist.all_reduce(op=SUM)
    #
    # So every GPU gets the GLOBAL totals.
    # --------------------------------------------------------

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

    # Final result is identical across all GPUs
    # (safe to log only on rank 0)
    return (
        running_loss / total,
        correct / total
    )