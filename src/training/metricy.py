import torch
import torch.distributed as dist


def reduce_sum(
    value,
    device
):

    tensor = torch.tensor(
        value,
        device=device
    )

    dist.all_reduce(
        tensor,
        op=dist.ReduceOp.SUM
    )

    return tensor.item()