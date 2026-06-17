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


# -------------------------
# Logging helper (DDP safe)
# -------------------------
def log(msg):
    if is_main_process():
        print(msg, flush=True)


def main():

    log("Starting training script...")

    local_rank = setup_ddp()
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

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=True)

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
                    "val_acc": val_acc
                },
                "pneumonia_model.pt"
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