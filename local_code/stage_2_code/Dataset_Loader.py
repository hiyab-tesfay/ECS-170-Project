"""Load the pre-split Stage 2 digit classification dataset."""

from pathlib import Path

import numpy as np

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    """Return train and test features and labels in the project format."""

    def __init__(self, dName="stage 2 digits", dDescription="", data_folder=None):
        super().__init__(dName, dDescription)
        self.dataset_source_folder_path = (
            Path(data_folder)
            if data_folder is not None
            else Path(__file__).resolve().parents[2] / "stage_2_data"
        )

    def _load_csv(self, file_name):
        path = self.dataset_source_folder_path / file_name
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing. If this is train.csv, run 'git lfs pull'."
            )

        rows = np.loadtxt(path, delimiter=",", dtype=np.int16, ndmin=2)
        if rows.shape[1] != 785:
            raise ValueError(f"{path} must have 785 columns; found {rows.shape[1]}")

        labels = rows[:, 0]
        pixels = rows[:, 1:]
        if np.any((labels < 0) | (labels > 9)):
            raise ValueError(f"{path} contains labels outside 0-9")
        if np.any((pixels < 0) | (pixels > 255)):
            raise ValueError(f"{path} contains pixel values outside 0-255")

        return {
            "X": pixels.astype(np.float32) / 255.0,
            "y": labels.astype(np.int64),
        }

    def load(self):
        """Load both fixed splits without shuffling or repartitioning them."""
        return {
            "train": self._load_csv("train.csv"),
            "test": self._load_csv("test.csv"),
        }
