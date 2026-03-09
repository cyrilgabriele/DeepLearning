import torch
import torch.nn as nn
import lightning as L
import numpy as np
from src.models.kan_layers import ChebyKANLayer, FourierKANLayer
from src.metrics.qwk import quadratic_weighted_kappa


class TabKAN(L.LightningModule):
    """Tabular KAN model with configurable layer type, depth, and width.

    Args:
        in_features: Number of input features.
        widths: List of hidden layer widths (determines depth).
        kan_type: "chebykan" or "fourierkan".
        degree: Chebyshev polynomial degree (only for chebykan).
        grid_size: Fourier grid size (only for fourierkan).
        lr: Learning rate.
        weight_decay: L2 regularization.
    """

    def __init__(
        self,
        in_features: int,
        widths: list[int],
        kan_type: str = "chebykan",
        degree: int = 3,
        grid_size: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        if kan_type not in ("chebykan", "fourierkan"):
            raise ValueError(f"Unknown kan_type: {kan_type}. Use 'chebykan' or 'fourierkan'.")

        layer_cls = ChebyKANLayer if kan_type == "chebykan" else FourierKANLayer
        layer_kwargs = {"degree": degree} if kan_type == "chebykan" else {"grid_size": grid_size}

        layers = []
        dims = [in_features] + widths
        for i in range(len(dims) - 1):
            layers.append(layer_cls(dims[i], dims[i + 1], **layer_kwargs))
            layers.append(nn.LayerNorm(dims[i + 1]))

        self.kan_layers = nn.Sequential(*layers)
        self.head = nn.Linear(widths[-1], 1)
        self.loss_fn = nn.MSELoss()

        self._val_preds = []
        self._val_targets = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.kan_layers(x)
        return self.head(h)

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
