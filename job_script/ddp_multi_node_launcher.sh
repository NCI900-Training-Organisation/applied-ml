#!/bin/bash

# Load shell environment variables
source ~/.bashrc

module load python3/3.11.0
module load cuda/11.7.0
module load intel-mkl/2022.2.0

source /scratch/vp91/pytorch-venv/bin/activate
 
export USER=$(whoami)

# Run PyTorch application
# ------------------------------------------------------------
# DISTRIBUTED TRAINING LAUNCH (torchrun - DDP)
# ------------------------------------------------------------
#
# torchrun is the official launcher for PyTorch Distributed
# Data Parallel (DDP) training. It starts multiple processes (one per GPU) and handles
# communication between them automatically.
#
# --nnodes=${1}
#   Number of compute nodes (machines) participating
#   in distributed training.
#
# --nproc_per_node=${2}
#   Number of processes per node.
#   Typically equals number of GPUs on that node.
#
#   Example:
#     4 GPUs → 4 processes
#
# --rdzv_id=100
#   Unique ID for this distributed job.
#   Used to identify and group all participating processes.
#
# --rdzv_backend=c10d
#   Rendezvous backend used for process coordination.
#   c10d is PyTorch’s default distributed communication backend.
#
# --rdzv_endpoint=${3}:29400
#   Address of the master node (rank 0) and port used
#   for process coordination (rendezvous server).
#
# script: /scratch/vp91/$USER/applied-ml/src/ddp/train.py
#
# This is the training script executed independently
# by each spawned process. Each process runs the same
# code but operates on a different GPU and data shard.
# ------------------------------------------------------------
torchrun \
  --nnodes=${1} \
  --nproc_per_node=${2} \
  --rdzv_id=100 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${3}:29400 \
  /scratch/vp91/$USER/applied-ml/src/ddp/train.py