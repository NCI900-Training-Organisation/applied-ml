import os

# Dataset

DATA_ROOT = "/g/data/vp91/Training-Data/chest_xray"

LABELS = [
    "PNEUMONIA",
    "NORMAL"
]

IMG_SIZE = 150

# Training

BATCH_SIZE = 32
NUM_WORKERS = 2
NUM_EPOCHS = 12

# Optimizer

LR = 1e-3

# Data limit from original notebook
MAX_IMAGES_PER_CLASS = 50

# Checkpoint path

SAVE_PATH = f"/scratch/vp91/{os.environ['USER']}/applied-ml/job_script/pneumonia_model_acc.pt"