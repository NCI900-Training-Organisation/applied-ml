import torch

from models.pneumonia_cnn import PneumoniaCNN
from data.dataset import PneumoniaDataset
from data.transforms import eval_transform
from torch.utils.data import DataLoader

from config import *


def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint.
    """

    # Create model architecture first
    model = PneumoniaCNN()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load weights into model
    model.load_state_dict(checkpoint["model"])

    # Move to device
    model = model.to(device)

    # Set evaluation mode
    model.eval()

    return model


def evaluate_model(model, loader, device):
    """
    Standard single-GPU evaluation loop.
    """

    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:

            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            preds = (torch.sigmoid(outputs) >= 0.5).float()

            correct += (preds == targets).sum().item()
            total += targets.size(0)

    acc = correct / total
    return acc


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Dataset / DataLoader
    # -------------------------
    test_dataset = PneumoniaDataset(
        f"{DATA_ROOT}/test",
        transform=eval_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # -------------------------
    # Load trained model
    # -------------------------
    model = load_model("best_model.pt", device)

    # -------------------------
    # Run evaluation
    # -------------------------
    test_acc = evaluate_model(model, test_loader, device)

    print(f"Test Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()