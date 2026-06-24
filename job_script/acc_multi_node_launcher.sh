#!/bin/bash

# Load shell environment variables
source ~/.bashrc

module load python3/3.11.0
module load cuda/11.7.0
module load intel-mkl/2022.2.0

source /scratch/vp91/pytorch-venv/bin/activate
 
export USER=$(whoami)

# ------------------------------------------------------------
# DISTRIBUTED TRAINING LAUNCH (Hugging Face Accelerate)
# ------------------------------------------------------------
#
# accelerate launch is a high-level launcher that abstracts
# away PyTorch DDP / multi-node / multi-GPU complexity.
#
# It automatically configures:
# - Process spawning (multi-GPU / multi-node)
# - Distributed backend (DDP under the hood)
# - Gradient synchronization
# - Mixed precision training
#
# ------------------------------------------------------------
# KEY DIFFERENCE vs torchrun:
# ------------------------------------------------------------
#
# torchrun  → low-level PyTorch DDP launcher
# accelerate → higher-level abstraction over DDP + Automatic Mixed Precision (AMP)
#
# accelerate internally still uses DDP (or FSDP etc.),
# but hides distributed setup details from user code.
#
#
# --num_processes $((${1} * ${2}))
#   Total number of processes across all machines.
#
#   = num_nodes × GPUs_per_node
#
#   Example:
#     2 nodes × 4 GPUs = 8 processes total
#
# --num_machines ${1}
#   Number of compute nodes participating in training.
#
# --main_process_ip ${3}
#   IP address of the main (rank 0) node.
#   Used for process coordination across machines.
#
# --machine_rank ${4}
#   Rank of the current machine.
#   Example:
#     0 = master node
#     1 = second node, etc.
#
# --main_process_port 29500
#   Network port used for inter-process communication.
#
# --rdzv_backend c10d
#   Rendezvous backend for coordinating processes.
#   (Same backend used by PyTorch distributed.)
#
# --mixed_precision fp16
#   Enables automatic mixed precision training:
#   - Faster computation
#   - Lower memory usage
#   - Uses FP16 for supported operations
#
# --dynamo_backend no
#   Disables TorchDynamo / graph compilation.
#   Keeps execution in eager mode (more stable/debuggable).
#
#
# script: /scratch/vp91/jxj900/applied-ml/atscale/accelerate/train.py
#
# This script runs independently on each process,
# but Accelerate ensures:
# - correct device placement
# - distributed coordination
# - gradient synchronization (DDP under the hood)
# ------------------------------------------------------------
accelerate launch \
    --num_processes $((${1} * ${2})) \
    --num_machines ${1} \
    --main_process_ip ${3} \
    --machine_rank ${4} \
    --main_process_port 29500 \
    --rdzv_backend c10d \
    --mixed_precision fp16 \
    --dynamo_backend no \
    /scratch/vp91/jxj900/applied-ml/atscale/accelerate/train.py