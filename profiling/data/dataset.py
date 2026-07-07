import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

from config import (
    LABELS,
    IMG_SIZE,
    MAX_IMAGES_PER_CLASS
)


class PneumoniaDataset(Dataset):

    def __init__(
        self,
        root_dir,
        transform=None
    ):

        self.transform = transform
        self.samples = []

        for label in LABELS:

            class_id = LABELS.index(label)

            folder = os.path.join(
                root_dir,
                label
            )

            images = os.listdir(folder)[
                :MAX_IMAGES_PER_CLASS
            ]

            for image_name in images:

                self.samples.append(
                    (
                        os.path.join(
                            folder,
                            image_name
                        ),
                        class_id
                    )
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(
        self,
        idx
    ):

        image_path, label = self.samples[idx]

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        image = image.astype(
            np.float32
        ) / 255.0

        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.float32
            )
        )