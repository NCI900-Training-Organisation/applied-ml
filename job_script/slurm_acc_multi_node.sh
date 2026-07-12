#!/bin/bash
#SBATCH -J acc_multi_train
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


source /ocean/projects/tra210016p/jjohn2/pytorch-venv/bin/activate

cd "$PROJECT/applied-ml"

NNODES=2
PROC_PER_NODE=8
NUM_PROCESSES=$((NNODES * PROC_PER_NODE))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500


echo "Master address    : ${MASTER_ADDR}"
echo "Master port       : ${MASTER_PORT}"
echo "Processes/node    : ${PROC_PER_NODE}"
echo "Total processes   : ${NUM_PROCESSES}"


srun --ntasks-per-node=1 bash -c '
echo "Host              : $(hostname)"
echo "Machine rank      : ${SLURM_NODEID}"
'

echo "Starting Accelerate training..."


# Run Hugging Face Accelerate application
# ------------------------------------------------------------
# DISTRIBUTED TRAINING LAUNCH (Accelerate)
# ------------------------------------------------------------
#
# Hugging Face Accelerate simplifies distributed training
# across multiple GPUs and compute nodes. It manages process
# spawning and distributed communication automatically.
#
# --num_processes ${NUM_PROCESSES}
#   Total number of training processes across all nodes.
#   Typically equals the total number of GPUs.
#
#   Example:
#     2 nodes × 4 GPUs → 8 processes
#
# --num_machines ${NNODES}
#   Total number of compute nodes (machines) participating
#   in distributed training.
#
# --main_process_ip ${MASTER_ADDR}
#   IP address or hostname of the main node (rank 0).
#   Used by all nodes to coordinate distributed training.
#
# --machine_rank ${SLURM_NODEID}
#   Unique rank assigned to the current compute node.
#   SLURM_NODEID provides the node rank automatically.
#
#   Example:
#     Node 1 → rank 0
#     Node 2 → rank 1
#
# --main_process_port ${MASTER_PORT}
#   Network port used for communication and coordination
#   between distributed training processes.
#
# --rdzv_backend c10d
#   Rendezvous backend used for process coordination.
#   c10d is PyTorch's distributed rendezvous backend.
#
# --mixed_precision fp16
#   Enables FP16 mixed-precision training.
#   This can reduce GPU memory usage and improve performance.
#
# --dynamo_backend no
#   Disables the PyTorch Dynamo compiler backend.
#
# ------------------------------------------------------------



srun accelerate launch \
    --num_processes ${NUM_PROCESSES} \
    --num_machines ${NNODES} \
    --main_process_ip ${MASTER_ADDR} \
    --machine_rank ${SLURM_NODEID} \
    --main_process_port ${MASTER_PORT} \
    --rdzv_backend c10d \
    --mixed_precision fp16 \
    --dynamo_backend no \
    atscale/accelerate/train.py

echo "Job done."
