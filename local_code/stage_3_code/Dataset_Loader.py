"""Load the instructor's fixed Stage 3 image splits."""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from local_code.base_class.dataset import dataset


SPECS = {
    "MNIST": ((28, 28), 0, 9, 60000, 10000),
    "ORL": ((112, 92, 3), 1, 40, 360, 40),
    "CIFAR": ((32, 32, 3), 0, 9, 50000, 10000),
}


class ImageSplit(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index].float().div(255.0), self.labels[index]


class Dataset_Loader(dataset):
    def __init__(self, name, data_folder=None):
        name = name.upper()
        if name not in SPECS:
            raise ValueError(f"Unknown dataset {name}; choose {tuple(SPECS)}")
        super().__init__(name, "Stage 3 image classification")
        self.name = name
        self.folder = Path(data_folder) if data_folder else Path(__file__).resolve().parents[2] / "stage_3_data"

    def load(self):
        path = self.folder / self.name
        if not path.is_file():
            raise FileNotFoundError(path)
        # Pickle can execute code: only use the instructor-provided local files.
        with path.open("rb") as stream:
            raw = pickle.load(stream)
        shape, low, high, n_train, n_test = SPECS[self.name]
        output = {}
        for split, expected in (("train", n_train), ("test", n_test)):
            records = raw[split]
            if len(records) != expected:
                raise ValueError(f"{self.name} {split}: expected {expected}, got {len(records)}")
            labels = np.asarray([r["label"] for r in records], dtype=np.int64)
            if np.any((labels < low) | (labels > high)):
                raise ValueError(f"{self.name} {split}: invalid label")
            images = np.stack([r["image"] for r in records])
            if images.shape[1:] != shape or not np.isfinite(images).all():
                raise ValueError(f"{self.name} {split}: invalid image shape or nonfinite values")
            if images.min() < 0 or images.max() > 255:
                raise ValueError(f"{self.name} {split}: pixels outside 0..255")
            if self.name == "ORL":
                if not np.array_equal(images[..., 0], images[..., 1]) or not np.array_equal(images[..., 0], images[..., 2]):
                    raise ValueError("ORL grayscale channels differ")
                images = images[..., 0]
                labels = labels - 1
            if self.name == "MNIST" and not np.array_equal(images, images.astype(np.uint8)):
                raise ValueError("MNIST pixels are not integer-valued")
            images = images.astype(np.uint8, copy=False)
            if images.ndim == 3:
                images = images[:, None, :, :]
            else:
                images = images.transpose(0, 3, 1, 2)
            output[split] = ImageSplit(np.ascontiguousarray(images), labels)
        return output
