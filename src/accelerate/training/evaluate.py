import torch


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    accelerator
):
    """
    Accelerate-aware evaluation function.

    Comparison with native DDP:
    ---------------------------

    Native DDP evaluation typically requires:

        images = images.to(device)
        targets = targets.to(device)

        dist.all_reduce(...)
        dist.all_gather(...)

    to combine metrics across processes.

    With Accelerate:

        - Device placement is handled automatically.
        - Distributed communication is hidden behind
          accelerator.gather().
        - The same code works on:
            * CPU
            * Single GPU
            * Multi-GPU
            * Multi-node clusters

    After accelerator.prepare(loader),
    each process receives a different shard of the dataset.
    """

    # Evaluation mode:
    # - disables dropout
    # - uses BatchNorm running statistics
    model.eval()

    # Local accumulators.
    #
    # Each process starts with metrics for its own
    # shard of the validation dataset.
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:

        # ----------------------------------------------------------
        # No manual device transfer required.
        #
        # DDP:
        #
        #     images = images.to(device)
        #     targets = targets.to(device)
        #
        # Accelerate:
        #
        #     accelerator.prepare(...)
        #
        # already placed batches on the correct device.
        # ----------------------------------------------------------

        outputs = model(images)

        loss = criterion(
            outputs,
            targets
        )

        # ----------------------------------------------------------
        # Loss accumulation
        #
        # Keep a weighted sum so we can compute a proper
        # dataset-wide average later.
        # ----------------------------------------------------------
        running_loss += (
            loss.detach()
            * images.size(0)
        )

        # Convert logits to binary predictions
        preds = (
            torch.sigmoid(outputs)
            >= 0.5
        ).float()

        # ----------------------------------------------------------
        # Metric synchronization
        #
        # DDP usually requires:
        #
        #     dist.all_gather(...)
        #
        # Accelerate provides:
        #
        #     accelerator.gather(...)
        #
        # which collects tensors from every process.
        #
        # Example:
        #
        # GPU0 predictions -> [1,0,1]
        # GPU1 predictions -> [0,1,1]
        #
        # gather() returns:
        #
        # [1,0,1,0,1,1]
        #
        # allowing computation of true global metrics.
        # ----------------------------------------------------------
        preds = accelerator.gather(preds)
        targets = accelerator.gather(targets)

        correct += (
            preds == targets
        ).sum().item()

        total += targets.numel()

    # --------------------------------------------------------------
    # Global loss aggregation
    #
    # running_loss currently contains the loss sum
    # for this process only.
    #
    # gather() collects the loss values from every process.
    # --------------------------------------------------------------
    running_loss = accelerator.gather(
        running_loss.unsqueeze(0)
    ).sum()

    # Dataset-wide average loss
    val_loss = (
        running_loss.item()
        / total
    )

    # Dataset-wide accuracy
    val_acc = (
        correct
        / total
    )

    return (
        val_loss,
        val_acc
    )