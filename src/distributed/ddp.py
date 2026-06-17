import os
import torch
import torch.distributed as dist


def setup_ddp():
    """
    Initialize Distributed Data Parallel (DDP) environment.

    In single-GPU training, you typically don't need any of this.
    In multi-GPU training, each GPU runs in a separate process,
    and we need to initialize a communication backend between them.
    """

    # Choose backend:
    # - "nccl" = best performance for NVIDIA GPUs (recommended for multi-GPU training)
    # - "gloo" = fallback for CPU or non-NVIDIA setups
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Initialize the default process group.
    # This is what enables communication (gradient sync, broadcasts, etc.)
    # between all GPU processes.
    dist.init_process_group(
        backend=backend
    )

    # Each process gets a LOCAL_RANK environment variable.
    # This tells us which GPU this process should use.
    # Example:
    #   rank 0 -> GPU 0
    #   rank 1 -> GPU 1
    #   rank 2 -> GPU 2 ...
    local_rank = int(os.environ["LOCAL_RANK"])

    # Bind this process to its assigned GPU.
    # Without this, all processes may try to use GPU 0 (bad in DDP).
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Return GPU index so model/data can be moved correctly later
    return local_rank


def cleanup_ddp():
    """
    Cleanly shut down the distributed process group.

    Important in multi-GPU runs to free communication resources.
    """
    dist.destroy_process_group()


def is_main_process():
    """
    In DDP, multiple processes run the same code.

    rank 0 is usually treated as the "main" process:
    - logs output
    - saves checkpoints
    - avoids duplicate printing
    """
    return dist.get_rank() == 0