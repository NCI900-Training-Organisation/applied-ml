import torch

from training.metrics import (
    reduce_sum
)


@torch.no_grad() # PyTorch decorator that disables gradient tracking for a function - no training, inference only
def evaluate(
    model,
    loader,
    criterion,
    device
):
    """
    DDP-aware evaluation function.

    In single-GPU training:
        - This runs once on the full dataset.
        - No communication needed.

    In multi-GPU (DDP) training:
        - Each GPU processes a different subset of the dataset.
        - Each process computes local metrics.
        - Metrics are then globally reduced using all-reduce (reduce_sum).
    """

    # EVALUATION MODE (IMPORTANT):
    # ----------------------------
    # Switches the model from training mode to inference
    # (evaluation) mode.
    #
    # This changes the behavior of certain layers:
    #
    # - Dropout is disabled so that all activations
    #   are used and predictions are deterministic.
    #
    # - BatchNorm stops updating running statistics
    #   (mean and variance) and instead uses the
    #   statistics learned during training.
    #
    # During validation/testing we want stable and
    # reproducible predictions, so model.eval() should
    # be called before the evaluation loop.
    #
    # In DDP, each GPU/process evaluates its local
    # model replica independently, but all replicas
    # are placed into evaluation mode.
    model.eval()

    # Local (per-GPU) accumulators
    # IMPORTANT: in DDP each process has its own copy of these
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:

        # Move batch to the GPU assigned to this process
        # In DDP each process has its own GPU (via local_rank)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass is independent per GPU
        outputs = model(images)

        # Compute loss locally (per batch, per GPU)
        loss = criterion(outputs, targets)

        # Accumulate *sum of loss over samples*
        # Multiply by batch size so final averaging is correct after reduction
        running_loss += loss.item() * images.size(0)

        # Convert logits → probabilities → binary predictions
        preds = (torch.sigmoid(outputs) >= 0.5).float()

        # Count correct predictions locally (per GPU)
        correct += (preds == targets).sum().item()

        # Count number of samples processed locally
        total += targets.size(0)

    # ---------------------------------------------------------------------
    # DDP SYNCHRONIZATION STEP (CRITICAL DIFFERENCE vs single GPU)
    # ---------------------------------------------------------------------
    #
    # At this point:
    #   - GPU 0 has partial metrics (its data shard)
    #   - GPU 1 has partial metrics
    #   - GPU 2 has partial metrics
    #
    # We must combine them into GLOBAL dataset-wide metrics.
    #
    # reduce_sum() is typically implemented using:
    #     dist.all_reduce(tensor, op=SUM)
    #
    # So every GPU ends up with the SAME global values.
    # ---------------------------------------------------------------------

    running_loss = reduce_sum(running_loss, device)
    correct = reduce_sum(correct, device)
    total = reduce_sum(total, device)

    # Final global metrics (identical on all GPUs)
    return (
        running_loss / total,   # average loss over full dataset
        correct / total         # accuracy over full dataset
    )