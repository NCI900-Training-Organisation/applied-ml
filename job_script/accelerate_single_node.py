#!/bin/bash

#PBS -P vp91
#PBS -q gpuvolta

#PBS -l ncpus=24
#PBS -l ngpus=2
#PBS -l mem=10GB
#PBS -l walltime=00:30:00 
#PBS -l storage=scratch/vp91+gdata/vp91

#PBS -N distributed_data_parallel

module load python3/3.11.0  
module load cuda/12.3.2

. /g/data/vp91/Training-Venvs/pytorch/bin/activate

accelerate launch --num_processes=2  /scratch/vp91/$USER/applied-ml/src/accelerate/train.py