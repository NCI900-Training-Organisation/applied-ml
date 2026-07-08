#!/bin/bash
#SBATCH -J acc_multi_train
#SBATCH -N 2
#SBATCH -p GPU
#SBATCH -t 00:15:00
#SBATCH --gpus=v100-32:16

set -euo pipefail

# Load modules
module purge
module load cuda/12.6.1

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export OMP_NUM_THREADS=4

# Activate Python environment
source /ocean/projects/tra210016p/jjohn2/pytorch-venv/bin/activate

# Move to project directory
cd $PROJECT/applied-ml

# Distributed training configuration
NNODES=${SLURM_NNODES}
PROC_PER_NODE=8
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "Starting multi-node Accelerate training..."
echo "Master node      : ${MASTER_ADDR}"
echo "Master port      : ${MASTER_PORT}"
echo "Number of nodes  : ${NNODES}"
echo "Processes/node   : ${PROC_PER_NODE}"
echo "Machine rank     : ${SLURM_NODEID}"

srun accelerate launch \
    --num_machines ${NNODES} \
    --machine_rank ${SLURM_NODEID} \
    --num_processes ${PROC_PER_NODE} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    atscale/accelerate/train.py

echo "Job done."