import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import lightning as L

from src.preprocessing import preprocess_kan_paper as kan_prep
from src.preprocessing.prudential_features import get_feature_lists


class TabularDataset(Dataset):
    """Simple dataset wrapping numpy arrays as float32 tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class _PreprocessorSummary:
    def __init__(self, feature_lists):
        self.feature_lists = feature_lists
        self.dropped_features: list[str] = []
        self.missing_threshold = 0.0


class PrudentialDataModule(L.LightningDataModule):
    """Lightning DataModule backed by the KAN-ready preprocessing pipeline."""

    def __init__(
        self,
        data_path: str,
        batch_size: int = 256,
        num_workers: int = 0,
        seed: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.preprocessor = None
        self.train_dataset = None
        self.val_dataset = None
        self._num_features = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.feature_names: list[str] | None = None

    @property
    def num_features(self) -> int:
        if self._num_features is None:
            raise RuntimeError("Call setup() first.")
        return self._num_features

    def setup(self, stage=None):
        raw_df = pd.read_csv(self.data_path)
        feature_lists = get_feature_lists(raw_df)
        self.preprocessor = _PreprocessorSummary(feature_lists)

        pipeline_outputs = kan_prep.run_pipeline(self.data_path, random_seed=self.seed)
        self.feature_names = pipeline_outputs["feature_names"]

        X_train = pipeline_outputs["X_train_outer"]
        y_train = pipeline_outputs["y_train_outer"].astype(np.float32)
        X_val = pipeline_outputs["X_test_outer"]
        y_val = pipeline_outputs["y_test_outer"].astype(np.float32)

        self._num_features = X_train.shape[1]
        self.train_dataset = TabularDataset(X_train, y_train)
        self.val_dataset = TabularDataset(X_val, y_val)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
