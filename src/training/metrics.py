import torch
import torch.distributed as dist


def reduce_sum(
    value,
    device
):
    """
    DDP utility function to aggregate a scalar value across all GPUs.

    In single-GPU training:
        - This function is effectively a no-op (no other processes exist).
        - The returned value is unchanged.

    In multi-GPU (DDP) training:
        - Each GPU/process computes its own local scalar (e.g., loss, correct, total).
        - We convert it into a tensor and perform an all-reduce SUM.
        - Every process receives the GLOBAL sum across all GPUs.

    Example:
        GPU0 value = 2
        GPU1 value = 3
        GPU2 value = 5

        After reduce_sum:
        All GPUs receive: 10
    """

    # Convert Python scalar into a GPU tensor
    # Each process creates its own local tensor
    tensor = torch.tensor(
        value,
        device=device
    )

    # Perform ALL-REDUCE across all processes in the process group
    # SUM means:
    #   - each GPU sends its tensor
    #   - all tensors are summed element-wise
    #   - result is written back to EVERY GPU
    dist.all_reduce(
        tensor,
        op=dist.ReduceOp.SUM #tells PyTorch what kind of reduction operation to perform during distributed communication.
    )

    # Convert back to Python scalar for logging / metrics
    return tensor.item()