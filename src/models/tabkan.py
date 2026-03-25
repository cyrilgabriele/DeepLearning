from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import lightning as L
import numpy as np
import pandas as pd

from src.models.kan_layers import ChebyKANLayer, FourierKANLayer, BSplineKANLayer
from src.metrics.qwk import quadratic_weighted_kappa
from src.models.base import PrudentialModel


class TabKAN(L.LightningModule):
    """Tabular KAN model with configurable layer type, depth, and width.

    Args:
        in_features: Number of input features.
        widths: List of hidden layer widths (determines depth).
        kan_type: "chebykan", "fourierkan", or "bsplinekan".
        degree: Chebyshev polynomial degree (only for chebykan).
        grid_size: Grid size (for fourierkan and bsplinekan).
        spline_order: B-spline order (only for bsplinekan).
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
        spline_order: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        sparsity_lambda: float = 0.0,
        l1_weight: float = 1.0,
        entropy_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        if kan_type not in ("chebykan", "fourierkan", "bsplinekan"):
            raise ValueError(f"Unknown kan_type: {kan_type}. Use 'chebykan', 'fourierkan', or 'bsplinekan'.")

        if kan_type == "chebykan":
            layer_cls = ChebyKANLayer
            layer_kwargs = {"degree": degree}
        elif kan_type == "fourierkan":
            layer_cls = FourierKANLayer
            layer_kwargs = {"grid_size": grid_size}
        else:
            layer_cls = BSplineKANLayer
            layer_kwargs = {"grid_size": grid_size, "spline_order": spline_order}

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

    def _sparsity_reg(self) -> torch.Tensor:
        """L1 + entropy regularisation over KAN activation functions.

        Implements the sparsity objective from Liu et al. (2024) arXiv:2404.19756 §2.5:
            ℓ_reg = λ · (μ₁ Σ_l ||Φ_l||₁ + μ₂ Σ_l S(Φ_l))
        where ||Φ_l||₁ is the summed L1 norm of all activation functions in layer l,
        and S(Φ_l) is the entropy of the distribution of edge magnitudes within layer l.
        L1 drives activations toward zero; entropy (minimised) produces a concentrated
        (sparse) distribution rather than many equally-small activations.
        """
        if self.hparams.sparsity_lambda == 0.0:
            return torch.zeros(1, device=self.device).squeeze()

        reg = torch.zeros(1, device=self.device).squeeze()
        for layer in self.kan_layers:
            if not isinstance(layer, (ChebyKANLayer, FourierKANLayer)):
                continue

            if isinstance(layer, ChebyKANLayer):
                # Mean absolute coefficient value per edge + base weight magnitude
                l1_per_edge = layer.cheby_coeffs.abs().mean(dim=-1) + layer.base_weight.abs()
            else:
                # Average over cosine and sine coefficients + base weight
                l1_per_edge = (
                    (layer.fourier_a.abs().mean(dim=-1) + layer.fourier_b.abs().mean(dim=-1)) / 2
                    + layer.base_weight.abs()
                )

            # L1 term: sum of all per-edge activation magnitudes
            l1_term = l1_per_edge.sum()

            # Entropy term: -Σ p_ij · log(p_ij), where p_ij = l1_ij / Σ l1
            total = l1_per_edge.sum() + 1e-10
            p = (l1_per_edge / total).clamp(min=1e-10)
            entropy_term = -(p * torch.log(p)).sum()

            reg = reg + self.hparams.l1_weight * l1_term + self.hparams.entropy_weight * entropy_term

        return self.hparams.sparsity_lambda * reg

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        reg = self._sparsity_reg()
        total_loss = loss + reg
        self.log("train/loss", loss, prog_bar=True)
        if self.hparams.sparsity_lambda > 0.0:
            self.log("train/sparsity_reg", reg, prog_bar=False)
        return total_loss

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


class TabKANClassifier(PrudentialModel):
    """Wraps TabKAN (Lightning) to expose the PrudentialModel fit/predict interface.

    This allows the Trainer pipeline from main.py to use the real TabKAN
    implementation instead of the sklearn placeholder.
    """

    PRESETS = {
        "tabkan-tiny": {"widths": (32, 16), "lr": 5e-3, "max_epochs": 50},
        "tabkan-small": {"widths": (64, 32), "lr": 3e-3, "max_epochs": 100},
        "tabkan-base": {"widths": (128, 64), "lr": 1e-3, "max_epochs": 100},
    }

    def __init__(
        self,
        preset: str = "tabkan-base",
        *,
        random_state: int = 42,
        flavor: str = "chebykan",
        depth: int | None = None,
        width: int | None = None,
        degree: int = 3,
        grid_size: int = 4,
        spline_order: int = 3,
        max_epochs: int | None = None,
        sparsity_lambda: float = 0.0,
        l1_weight: float = 1.0,
        entropy_weight: float = 1.0,
        **extra_params,
    ) -> None:
        params = {
            "preset": preset, "flavor": flavor, "depth": depth,
            "width": width, "degree": degree, "random_state": random_state,
        }
        params.update(extra_params)
        super().__init__(**params)

        base = self.PRESETS.get(preset, self.PRESETS["tabkan-base"])
        self.kan_type = flavor
        self.degree = degree
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.random_state = random_state
        self.max_epochs = max_epochs or base["max_epochs"]
        self.lr = base["lr"]
        self.sparsity_lambda = sparsity_lambda
        self.l1_weight = l1_weight
        self.entropy_weight = entropy_weight

        base_widths = list(base["widths"])
        if depth is not None and width is not None:
            self.widths = [width] * depth
        elif depth is not None:
            self.widths = [base_widths[0]] * depth
        elif width is not None:
            self.widths = [width] * len(base_widths)
        else:
            self.widths = base_widths

        self.module: TabKAN | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        validation_data: Tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> None:
        L.seed_everything(self.random_state)

        X_values = self._to_numpy_features(X)
        y_values = self._to_numpy_targets(y)

        in_features = X_values.shape[1]
        self.module = TabKAN(
            in_features=in_features,
            widths=self.widths,
            kan_type=self.kan_type,
            degree=self.degree,
            grid_size=self.grid_size,
            spline_order=self.spline_order,
            lr=self.lr,
            sparsity_lambda=self.sparsity_lambda,
            l1_weight=self.l1_weight,
            entropy_weight=self.entropy_weight,
        )

        X_t = torch.tensor(X_values, dtype=torch.float32)
        y_t = torch.tensor(y_values, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t, y_t.unsqueeze(1))
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_t = torch.tensor(self._to_numpy_features(X_val), dtype=torch.float32)
            y_val_t = torch.tensor(self._to_numpy_targets(y_val), dtype=torch.float32)
            val_ds = torch.utils.data.TensorDataset(X_val_t, y_val_t.unsqueeze(1))
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False)

        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            deterministic=True,
        )
        fit_kwargs = {"train_dataloaders": loader}
        if val_loader is not None:
            fit_kwargs["val_dataloaders"] = val_loader
        trainer.fit(self.module, **fit_kwargs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.module is None:
            raise RuntimeError("Call fit() before predict().")

        self.module.eval()
        X_t = torch.tensor(self._to_numpy_features(X), dtype=torch.float32)
        with torch.no_grad():
            preds = self.module(X_t).cpu().numpy().flatten()
        return np.clip(np.round(preds), 1, 8).astype(int)

    @staticmethod
    def _to_numpy_features(X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float32, copy=False)
        return np.asarray(X, dtype=np.float32)

    @staticmethod
    def _to_numpy_targets(y: pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(y, pd.Series):
            return y.to_numpy(dtype=np.float32, copy=False)
        return np.asarray(y, dtype=np.float32)


def build_tabkan_model(
    preset: str,
    *,
    random_state: int,
    flavor: str | None = None,
    depth: int | None = None,
    width: int | None = None,
    degree: int | None = None,
    **extra_params,
) -> TabKANClassifier:
    """Factory function for the model registry."""
    kwargs: dict = {"random_state": random_state}
    if flavor is not None:
        kwargs["flavor"] = flavor
    if depth is not None:
        kwargs["depth"] = depth
    if width is not None:
        kwargs["width"] = width
    if degree is not None:
        kwargs["degree"] = degree
    kwargs.update(extra_params)
    return TabKANClassifier(preset, **kwargs)
