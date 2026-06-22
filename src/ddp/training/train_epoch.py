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

    # Put model into training mode.
    #
    # This enables training-specific layer behavior:
    # - Dropout randomly drops activations.
    # - BatchNorm updates running statistics.
    #
    # In DDP, every process/GPU calls model.train() on its
    # local model replica so all replicas operate in
    # training mode.
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

        # CLEAR GRADIENTS (IMPORTANT):
        # ----------------------------
        # PyTorch accumulates gradients by default.
        #
        # This means that every call to loss.backward()
        # ADDS new gradients to the existing values in
        # parameter.grad rather than replacing them.
        #
        # Example:
        #
        #   Iteration 1:
        #     gradient = 2
        #
        #   Iteration 2:
        #     gradient = 3
        #
        # Without optimizer.zero_grad():
        #
        #     parameter.grad = 2 + 3 = 5
        #
        # With optimizer.zero_grad():
        #
        #     Iteration 1 -> parameter.grad = 2
        #     clear gradients
        #     Iteration 2 -> parameter.grad = 3
        #
        # We therefore clear gradients before processing
        # each mini-batch so that the backward pass computes
        # gradients only for the CURRENT batch.
        #
        # In DDP, each GPU/process clears the gradients of
        # its local model replica before computing the next
        # set of gradients.
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
        #
        # Example:
        #   GPU0 gradient = 4
        #   GPU1 gradient = 8
        #
        # DDP performs an AllReduce and averages:
        #
        #   (4 + 8) / 2 = 6
        #
        # Both GPUs then have:
        #
        #   GPU0 gradient = 6
        #   GPU1 gradient = 6
        #
        # The optimizer on every GPU therefore applies the
        # exact same parameter update, keeping all model
        # replicas synchronized.
        loss.backward()

        # Apply synchronized gradients
        # Apply the parameter update.
        #
        # At this point:
        # - loss.backward() has computed gradients.
        # - DDP has already synchronized those gradients
        #   across all GPUs via AllReduce.
        #
        # optimizer.step() uses the synchronized gradients
        # to update the model parameters:
        #
        #   weight = weight - learning_rate * gradient
        #
        # Because every GPU has identical gradients,
        # every process performs the same update and
        # all model replicas remain identical.
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