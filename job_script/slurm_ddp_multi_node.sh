#!/bin/bash
#SBATCH -J ddp_multi_train
#SBATCH -N 2
#SBATCH -p GPU
#SBATCH -t 00:15:00
#SBATCH --gpus=v100-32:16

set -euo pipefail

module purge
module load cuda/12.6.1

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export OMP_NUM_THREADS=4

# Activate your Python environment
source /ocean/projects/tra210016p/jjohn2/pytorch-venv/bin/activate

# Move to your project directory
cd "$PROJECT/applied-ml"

NNODES=2
PROC_PER_NODE=8

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29400

echo "Master address    : ${MASTER_ADDR}"
echo "Master port       : ${MASTER_PORT}"
echo "Number of nodes   : ${NNODES}"
echo "Processes/node    : ${PROC_PER_NODE}"

srun --ntasks-per-node=1 bash -c '
echo "Host              : $(hostname)"
echo "Node rank         : ${SLURM_NODEID}"
'

echo "Starting DDP training..."

# Run PyTorch DDP application
# ------------------------------------------------------------
# DISTRIBUTED TRAINING LAUNCH (torchrun - DDP)
# ------------------------------------------------------------
#
# srun starts one torchrun launcher on each compute node.
#
# torchrun then starts one training process per GPU on the
# current node.
#
# --nnodes ${NNODES}
#   Total number of compute nodes participating in training.
#
#   Example:
#     2 nodes
#
# --nproc_per_node ${PROC_PER_NODE}
#   Number of training processes started on each node.
#   Typically equals the number of GPUs per node.
#
#   Example:
#     8 GPUs/node -> 8 processes/node
#
# --node_rank ${SLURM_NODEID}
#   Unique rank assigned to the current compute node.
#   SLURM_NODEID provides the node rank automatically.
#
#   Example:
#     Node 1 -> rank 0
#     Node 2 -> rank 1
#
# --rdzv_id ${SLURM_JOB_ID}
#   Unique identifier for the distributed training job.
#   The Slurm job ID prevents rendezvous conflicts between jobs.
#
# --rdzv_backend c10d
#   Uses PyTorch's c10d rendezvous backend for distributed
#   process coordination.
#
# --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT}
#   Address of the rendezvous node.
#   All nodes connect to this address before training starts.
#
# ------------------------------------------------------------

srun --ntasks-per-node=1 torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${PROC_PER_NODE} \
    --node_rank=${SLURM_NODEID} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    atscale/ddp/train.py

echo "Job done."