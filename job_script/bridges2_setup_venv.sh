#!/bin/bash

# Load required modules
module load cuda/12.6.1
module load python/3.8.6

# Activate the virtual environment
source /ocean/projects/tra210016p/jjohn2/pytorch-venv/bin/activate

# Register the Jupyter kernel
python3 -m ipykernel install \
    --user \
    --name ihpcss \
    --display-name "ihpcss"

echo "Kernel 'ihpcss' has been registered."
