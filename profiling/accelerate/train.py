from accelerate import Accelerator

import torch
import torch.nn as nn
import torch.optim as optim

from config import *

from models.pneumonia_cnn import PneumoniaCNN
from data.dataset import PneumoniaDataset
from data.transforms import train_transform, eval_transform

from training.train_epoch import train_epoch
from training.evaluate import evaluate

from torch.utils.data import DataLoader

from config import (
    SAVE_PATH
)

# -------------------------
# Logging helper (DDP safe)
# -------------------------
def log(accelerator, msg):
    if accelerator.is_main_process:
        print(msg, flush=True)


def main():

    accelerator = Accelerator(
        mixed_precision="fp16"
    )

    log(accelerator, "Starting training job")
    log(accelerator, f"Number of processes: {accelerator.num_processes}")
    log(accelerator, f"Mixed precision: {accelerator.mixed_precision}")
    log(accelerator, "Creating datasets")

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

    log(accelerator,
        f"Dataset sizes | "
        f"train={len(train_dataset)} "
        f"val={len(val_dataset)} "
        f"test={len(test_dataset)}"
    )

    log(accelerator, "Creating dataloaders")

    # DATA LOADERS (IMPORTANT NOTE FOR ACCELERATE):
    # --------------------------------------------
    # Unlike native DDP setups, Accelerate does NOT
    # require a DistributedSampler or a special
    # distributed DataLoader.
    #
    # In standard DDP, you typically must do:
    #
    #   sampler = DistributedSampler(dataset)
    #   DataLoader(dataset, sampler=sampler, ...)
    #
    # so that each GPU sees a unique shard of data.
    #
    # With Accelerate, this is handled automatically
    # when you call:
    #
    #   accelerator.prepare(...)
    #
    # Accelerate internally wraps the DataLoader and
    # applies the correct distributed sampling logic
    # per process.
    #
    # This means you can define normal PyTorch
    # DataLoaders as usual:
    #
    # - shuffle=True for training (handled per process)
    # - shuffle=False for validation/testing
    #
    # and Accelerate ensures each GPU receives its
    # own non-overlapping batch subset during training.
    #
    # So no explicit DistributedSampler is required
    # in user code when using Accelerate.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    log(accelerator,
        f"Dataloaders ready | "
        f"batch_size={BATCH_SIZE}"
    )

    log(accelerator, "Initializing model")

    model = PneumoniaCNN()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR
    )

    criterion = nn.BCEWithLogitsLoss()

    log(accelerator, f"Optimizer: Adam (lr={LR})")
    log(accelerator, "Preparing objects with Accelerate")

    model, optimizer, train_loader, val_loader = (
        accelerator.prepare(
            model,
            optimizer,
            train_loader,
            val_loader
        )
    )

    log(accelerator, "Accelerate preparation complete")

    best_val_acc = 0.0

    log(accelerator, f"Beginning training for {NUM_EPOCHS} epochs")

    for epoch in range(NUM_EPOCHS):

        log(accelerator, f"Epoch {epoch + 1}/{NUM_EPOCHS} started")

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            accelerator
        )

        log(accelerator,
            f"Epoch {epoch + 1}: "
            f"training complete "
            f"(loss={train_loss:.4f}, acc={train_acc:.4f})"
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            accelerator
        )

        log(accelerator,
            f"Epoch {epoch + 1}: "
            f"validation complete "
            f"(loss={val_loss:.4f}, acc={val_acc:.4f})"
        )

        if accelerator.is_main_process:

            print(
                f"Epoch {epoch + 1}: "
                f"Train={train_loss:.4f} "
                f"Val={val_loss:.4f} "
                f"Acc={val_acc:.4f}",
                flush=True
            )

            if val_acc > best_val_acc:

                log(accelerator,
                    f"Validation accuracy improved "
                    f"{best_val_acc:.4f} -> {val_acc:.4f}"
                )

                best_val_acc = val_acc

                unwrapped_model = accelerator.unwrap_model(model)

                torch.save(
                    unwrapped_model.state_dict(),
                    SAVE_PATH
                )

                log(accelerator, "Saved checkpoint: best_model.pt")

        log(accelerator, f"Epoch {epoch + 1}/{NUM_EPOCHS} finished")

    log(accelerator, "Training completed")
    log(accelerator, f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()