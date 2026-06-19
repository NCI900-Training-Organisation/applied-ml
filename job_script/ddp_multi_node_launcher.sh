#!/bin/bash

# Load shell environment variables
source ~/.bashrc

module load python3/3.11.0
module load cuda/11.7.0
module load intel-mkl/2022.2.0

source /scratch/vp91/pytorch-venv/bin/activate
 
export USER=$(whoami)

# Run PyTorch application
torchrun \
  --nnodes=${1} \
  --nproc_per_node=${2} \
  --rdzv_id=100 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${3}:29400 \
  /scratch/vp91/$USER/applied-ml/src/ddp/train.py