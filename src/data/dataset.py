import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import lightning as L
from sklearn.model_selection import StratifiedShuffleSplit
from src.data.sota_preprocessing import SOTAPreprocessor


class TabularDataset(Dataset):
    """Simple dataset wrapping numpy arrays as float32 tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PrudentialDataModule(L.LightningDataModule):
    """Lightning DataModule for Prudential Life Insurance dataset.

    Wraps SOTAPreprocessor for preprocessing, creates stratified
    train/val splits, and serves DataLoaders.
    """

    def __init__(
        self,
        data_path: str,
        val_split: float = 0.2,
        batch_size: int = 256,
        num_workers: int = 0,
        use_sota: bool = True,
        missing_threshold: float = 0.5,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.preprocessor = SOTAPreprocessor(
            missing_threshold=missing_threshold, use_sota=use_sota
        )
        self.train_dataset = None
        self.val_dataset = None
        self._num_features = None

    @property
    def num_features(self) -> int:
        if self._num_features is None:
            raise RuntimeError("Call setup() first.")
        return self._num_features

    def setup(self, stage=None):
        df = pd.read_csv(self.data_path)
        y = df["Response"].values
        X = df.drop(columns=["Response"])

        X_processed = self.preprocessor.fit_transform(X, y)
        X_np = X_processed.values.astype(np.float32)
        self._num_features = X_np.shape[1]

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=self.val_split, random_state=self.seed
        )
        train_idx, val_idx = next(sss.split(X_np, y))

        self.train_dataset = TabularDataset(X_np[train_idx], y[train_idx].astype(np.float32))
        self.val_dataset = TabularDataset(X_np[val_idx], y[val_idx].astype(np.float32))

        self.X_train = X_np[train_idx]
        self.y_train = y[train_idx].astype(np.float32)
        self.X_val = X_np[val_idx]
        self.y_val = y[val_idx].astype(np.float32)

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
