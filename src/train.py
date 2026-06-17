import os

import torch
import torch.nn as nn

from torch.nn.parallel import (
    DistributedDataParallel as DDP
)

from torch.utils.data import (
    DataLoader
)

from torch.utils.data.distributed import (
    DistributedSampler
)

from torch.optim.lr_scheduler import (
    ReduceLROnPlateau
)

from config import *

from distributed.ddp import (
    setup_ddp,
    cleanup_ddp,
    is_main_process
)

from models.pneumonia_cnn import (
    PneumoniaCNN
)

from data.dataset import (
    PneumoniaDataset
)

from data.transforms import (
    train_transform,
    eval_transform
)

from training.train_epoch import (
    train_epoch
)

from training.evaluate import (
    evaluate
)


def main():

    local_rank = setup_ddp()

    device = torch.device(
        f"cuda:{local_rank}"
    )

    train_dataset = PneumoniaDataset(
        os.path.join(
            DATA_ROOT,
            "train"
        ),
        transform=train_transform
    )

    val_dataset = PneumoniaDataset(
        os.path.join(
            DATA_ROOT,
            "val"
        ),
        transform=eval_transform
    )

    test_dataset = PneumoniaDataset(
        os.path.join(
            DATA_ROOT,
            "test"
        ),
        transform=eval_transform
    )

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

    model = PneumoniaCNN().to(
        device
    )

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank
    )

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

    for epoch in range(NUM_EPOCHS):

        train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_acc)

        if is_main_process():

            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                f"Train Loss={train_loss:.4f} "
                f"Train Acc={train_acc:.4f} "
                f"Val Loss={val_loss:.4f} "
                f"Val Acc={val_acc:.4f}"
            )

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    if is_main_process():

        print(
            f"Test Loss={test_loss:.4f}"
        )

        print(
            f"Test Accuracy={test_acc*100:.2f}%"
        )

    cleanup_ddp()


if __name__ == "__main__":
    main()