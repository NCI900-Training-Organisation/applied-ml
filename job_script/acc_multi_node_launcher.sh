#!/bin/bash

# Load shell environment variables
source ~/.bashrc

module load python3/3.11.0
module load cuda/11.7.0
module load intel-mkl/2022.2.0

source /scratch/vp91/pytorch-venv/bin/activate
 
export USER=$(whoami)

 # Run PyTorch application
accelerate launch \
    --num_processes $((${1} * ${2})) \
    --num_machines ${1} \
    --main_process_ip ${3} \
    --machine_rank ${4} \
    --main_process_port 29500 \
    --rdzv_backend c10d \
    --mixed_precision fp16 \
    --dynamo_backend no \
    /scratch/vp91/jxj900/applied-ml/src/accelerate/train.py