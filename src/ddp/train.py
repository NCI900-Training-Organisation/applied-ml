import os

import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import *

from distributed.ddp import setup_ddp, cleanup_ddp, is_main_process

from models.pneumonia_cnn import PneumoniaCNN
from data.dataset import PneumoniaDataset
from data.transforms import train_transform, eval_transform

from training.train_epoch import train_epoch
from training.evaluate import evaluate

from config import (
    SAVE_PATH
)

# -------------------------
# Logging helper (DDP safe)
# -------------------------
def log(msg):
    if is_main_process():
        print(msg, flush=True)


def main():

    local_rank = setup_ddp()

    log("Starting training script...")
    log(f"DDP initialized | local_rank={local_rank}")

    device = torch.device(f"cuda:{local_rank}")
    log(f"Device set to {device}")

    # -------------------------
    # Dataset loading
    # -------------------------
    log("Loading datasets...")

    train_dataset = PneumoniaDataset(
        os.path.join(DATA_ROOT, "train"),
        transform=train_transform
    )

    val_dataset = PneumoniaDataset(
        os.path.join(DATA_ROOT, "val"),
        transform=eval_transform
    )

    test_dataset = PneumoniaDataset(
        os.path.join(DATA_ROOT, "test"),
        transform=eval_transform
    )

    log(
        f"Dataset sizes | "
        f"train={len(train_dataset)} "
        f"val={len(val_dataset)} "
        f"test={len(test_dataset)}"
    )

    # -------------------------
    # Samplers
    # -------------------------
    log("Creating DistributedSamplers...")

    # DISTRIBUTED SAMPLERS (DDP):
    # --------------------------
    # In native PyTorch DDP, each process (GPU) must see
    # a UNIQUE subset of the dataset.
    #
    # DistributedSampler handles this by splitting the
    # dataset across all processes.
    #
    # Example with 4 GPUs:
    #
    #   GPU0 -> shard 0
    #   GPU1 -> shard 1
    #   GPU2 -> shard 2
    #   GPU3 -> shard 3
    #
    # Without this, every GPU would see the FULL dataset,
    # leading to duplicated computation and incorrect
    # gradient scaling.
    #
    # shuffle=True (train only):
    #   Ensures each epoch reshuffles data differently
    #   across all shards.
    #
    # shuffle=False (val/test):
    #   Keeps evaluation deterministic and consistent.
    #
    # drop_last=True:
    #   Drops leftover samples that cannot be evenly
    #   divided across GPUs.
    #
    #   This ensures all GPUs receive equal batch sizes,
    #   preventing synchronization issues during DDP
    #   training (especially for loss.backward()).
    #
    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        drop_last=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        shuffle=False,
        drop_last=True
    )
    
    test_sampler = DistributedSampler(
        test_dataset,
        shuffle=False,
        drop_last=True
    )

    # -------------------------
    # DataLoaders
    # -------------------------
    log("Creating DataLoaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        sampler=test_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    log("DataLoaders ready")

    # -------------------------
    # Model
    # -------------------------
    log("Initializing model...")

    model = PneumoniaCNN().to(device)

    log("Wrapping model with DDP...")

    # WRAP MODEL WITH DDP (IMPORTANT):
    # --------------------------------
    # DDP (DistributedDataParallel) creates one model
    # replica per process/GPU and handles gradient
    # synchronization automatically.
    #
    # This ensures all model replicas remain identical
    # across GPUs.
    #
    # device_ids specifies which GPU this process owns.
    # For example:
    #
    #   local_rank = 0 → GPU0
    #   local_rank = 1 → GPU1
    #   local_rank = 2 → GPU2
    #
    # Each process only performs computation on its
    # assigned GPU.
    #
    # output_device specifies where outputs from the
    # forward pass are placed. Typically this is the
    # same GPU assigned to the process.
    #
    # Example:
    #
    #   Process 0 → GPU0 → device_ids=[0]
    #   Process 1 → GPU1 → device_ids=[1]
    #   Process 2 → GPU2 → device_ids=[2]
    #
    # DDP does NOT split batches across GPUs. Instead,
    # each process receives a different subset of data
    # (usually via DistributedSampler) and trains its
    # own model replica. Gradients are synchronized
    # automatically during loss.backward().
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank
    )

    log("Model ready in DDP mode")

    # -------------------------
    # Loss / Optim / Scheduler
    # -------------------------
    log("Setting up loss, optimizer, scheduler...")

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=LR
    )

    # Reduce learning rate when validation metric
    # stops improving.
    #
    # mode="max"     -> larger metric values are better
    # factor=0.3     -> new_lr = old_lr * 0.3
    # patience=2     -> wait 2 epochs before reducing
    # min_lr=1e-6    -> lower learning-rate limit
    #
    # Typically called as:
    #   scheduler.step(val_metric)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.3,
        patience=2,
        min_lr=1e-6
    )

    log("Training components initialized")

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(NUM_EPOCHS):

        log(f"Epoch {epoch+1}/{NUM_EPOCHS} starting")

        train_sampler.set_epoch(epoch)
        log("Sampler epoch set")

        log("Training started")
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        log(f"Train done | loss={train_loss:.4f}, acc={train_acc:.4f}")

        log("Validation started")
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        log(f"Validation done | loss={val_loss:.4f}, acc={val_acc:.4f}")

        scheduler.step(val_acc)

        log("Saving checkpoint")

        if is_main_process():
            

            torch.save(
                {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                SAVE_PATH,
            )

        log("Checkpoint saved")

    # -------------------------
    # Final evaluation
    # -------------------------
    log("Running final test evaluation")

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    if is_main_process():
        print(f"Test Loss = {test_loss:.4f}")
        print(f"Test Accuracy = {test_acc * 100:.2f}%")

    log("Cleaning up DDP")

    cleanup_ddp()

    log("Training complete")


if __name__ == "__main__":
    main()