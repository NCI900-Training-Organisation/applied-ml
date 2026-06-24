import torch


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    accelerator
):
    """
    Accelerate training epoch.

    Comparison with native DDP:
    ---------------------------

    Native DDP workflow typically requires:

        model = DDP(model, device_ids=[local_rank])

        images = images.to(device)
        targets = targets.to(device)

        loss.backward()

        dist.all_reduce(...)
        dist.barrier(...)
        DistributedSampler(...)

    Hugging Face Accelerate hides most of this complexity.

    After calling:

        model, optimizer, loader = accelerator.prepare(...)

    Accelerate automatically:

    - Places the model on the correct device.
    - Places batches on the correct device.
    - Wraps the model with DDP when multiple GPUs exist.
    - Handles gradient synchronization internally.
    - Configures distributed execution.
    - Works on CPU, single GPU, multi-GPU, and multi-node
      with the same code.

    The main differences you'll see in the training loop are:

    1. Use accelerator.backward(loss)
       instead of loss.backward()

    2. Use accelerator.gather(...)
       instead of manual dist.all_gather/all_reduce
       for metrics.

    3. No explicit device management.
    """

    # Put model into training mode
    model.train()

    # Local metric accumulators.
    #
    # Each process/GPU maintains its own copy initially.
    running_loss = 0.0
    correct = 0
    total = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    #
    # With Accelerate:
    #
    # - Each GPU receives a different shard of the dataset.
    # - Data is already on the correct device.
    # - No need for images.to(device).
    #
    # In native DDP you would usually move tensors manually.
    # ------------------------------------------------------------------
    for images, targets in loader:

        # Clear gradients for this optimizer
        optimizer.zero_grad()

        # Forward pass on the local GPU
        outputs = model(images)

        # Loss for this GPU's mini-batch
        loss = criterion(
            outputs,
            targets
        )

        # BACKWARD PASS (ACCELERATE):
        # ---------------------------
        # Native PyTorch DDP typically uses:
        #
        #     loss.backward()
        #
        # Accelerate replaces this with:
        #
        #     accelerator.backward(loss)
        #
        # This provides a single API that works for:
        # - Single GPU
        # - Multi-GPU (DDP)
        # - Mixed precision (fp16/bf16)
        #
        # When running with multiple GPUs, Accelerate
        # still uses DDP under the hood and gradients are
        # synchronized across GPUs during the backward pass.
        #
        # The main difference is that Accelerate manages
        # backend-specific details (such as gradient scaling
        # for mixed precision) so the training code does not
        # need to change between hardware configurations.
        accelerator.backward(loss)

        # Apply synchronized gradients
        optimizer.step()

        # --------------------------------------------------------------
        # Loss accumulation
        # --------------------------------------------------------------
        #
        # Multiply by batch size so we can compute a proper
        # sample-weighted average at the end.
        #
        # loss.item() would also work, but keeping this as a tensor
        # makes later distributed gathering easier.
        # --------------------------------------------------------------
        running_loss += (
            loss.detach()
            * images.size(0)
        )

        # Convert logits -> binary predictions
        preds = (
            torch.sigmoid(outputs)
            >= 0.5
        ).float()

        # METRIC SYNCHRONIZATION (ACCELERATE):
        # ------------------------------------
        # During multi-GPU training, each process only sees
        # its own subset of data and therefore only has
        # partial metrics.
        #
        # Native DDP typically requires explicit collective
        # communication calls such as:
        #
        #     dist.all_gather(...)
        #     dist.all_reduce(...)
        #
        # to combine results across GPUs.
        #
        # Accelerate provides:
        #
        #     accelerator.gather(...)
        #
        # which collects tensors from all processes using a
        # simpler, backend-independent API.
        #
        # Example:
        #
        #     GPU0 -> 256 samples
        #     GPU1 -> 256 samples
        #     GPU2 -> 256 samples
        #     GPU3 -> 256 samples
        #
        # After gather(), metrics can be computed using the
        # combined results from all 1024 samples.
        preds = accelerator.gather(preds)
        targets = accelerator.gather(targets)

        # Accuracy is now computed using predictions from ALL GPUs
        correct += (
            preds == targets
        ).sum().item()

        # Total number of samples across all GPUs
        total += targets.numel()

    # ------------------------------------------------------------------
    # Global loss aggregation
    # ------------------------------------------------------------------
    #
    # running_loss currently contains only the local process value.
    #
    # accelerator.gather() collects one loss value from every GPU.
    #
    # Example:
    #
    # GPU0 -> 12.5
    # GPU1 -> 11.8
    # GPU2 -> 13.1
    # GPU3 -> 12.0
    #
    # gather -> [12.5, 11.8, 13.1, 12.0]
    #
    # sum -> 49.4
    #
    # This becomes the global loss numerator.
    # ------------------------------------------------------------------
    running_loss = accelerator.gather(
        running_loss.unsqueeze(0)
    ).sum()

    # Global average loss
    train_loss = (
        running_loss.item()
        / total
    )

    # Global accuracy
    train_acc = (
        correct
        / total
    )

    return train_loss, train_acc