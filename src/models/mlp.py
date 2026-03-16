import torch
import torch.nn as nn
import lightning as L
import numpy as np
from src.metrics.qwk import quadratic_weighted_kappa


class MLPBaseline(L.LightningModule):
    """Standard MLP baseline for tabular regression.

    Args:
        in_features: Number of input features.
        widths: List of hidden layer widths.
        dropout: Dropout probability between layers.
        lr: Learning rate.
        weight_decay: L2 regularization.
    """

    def __init__(
        self,
        in_features: int,
        widths: list[int],
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        dims = [in_features] + widths
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(dims[i + 1]))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(widths[-1], 1)
        self.loss_fn = nn.MSELoss()

        self._val_preds = []
        self._val_targets = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val/loss", loss, prog_bar=True)
        self._val_preds.append(y_hat.detach().cpu().numpy())
        self._val_targets.append(y.detach().cpu().numpy())
        return loss

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        preds = np.concatenate(self._val_preds).flatten()
        targets = np.concatenate(self._val_targets).flatten()
        preds_rounded = np.clip(np.round(preds), 1, 8).astype(int)
        targets_int = np.clip(np.round(targets), 1, 8).astype(int)
        kappa = quadratic_weighted_kappa(targets_int, preds_rounded)
        self.log("val/qwk", kappa, prog_bar=True)
        self._val_preds.clear()
        self._val_targets.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
