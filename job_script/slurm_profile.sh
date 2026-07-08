#!/bin/bash
#SBATCH -J profile_accelerate
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 00:30:00
#SBATCH --gpus=v100-32:2

set -euo pipefail

module purge
module load cuda/12.6.1

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4

# Activate your Python environment
source /ocean/projects/tra210016p/jjohn2/pytorch-venv/bin/activate

# Move to your project directory
cd $PROJECT/applied-ml

echo "Starting Accelerate training..."

accelerate launch \
    --num_processes 2 \
    profiling/train.py

echo "Job done."