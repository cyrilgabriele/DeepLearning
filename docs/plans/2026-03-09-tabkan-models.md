# TabKAN Model Training Framework — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete training framework for ChebyKAN and FourierKAN tabular models with MLP and XGBoost baselines, Hydra configuration, and QWK-optimized evaluation.

**Architecture:** Hydra-configured PyTorch Lightning pipeline. KAN layers (ChebyKAN, FourierKAN) are swappable `nn.Module`s plugged into a shared `TabKAN` LightningModule. Data flows through the existing `SOTAPreprocessor`, wrapped in a LightningDataModule. All models output a single regression scalar; ordinal predictions are recovered via threshold optimization maximizing QWK.

**Tech Stack:** PyTorch, PyTorch Lightning, Hydra, XGBoost, scipy, scikit-learn, TensorBoard

---

## Task 1: Add Dependencies to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add all required packages**

Update `pyproject.toml` dependencies to:

```toml
[project]
name = "parrotlabs"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "category-encoders>=2.9.0",
    "matplotlib>=3.10.8",
    "pandas>=3.0.1",
    "torch>=2.6.0",
    "lightning>=2.5.0",
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "xgboost>=2.1.0",
    "scikit-learn>=1.6.0",
    "scipy>=1.15.0",
    "tensorboard>=2.19.0",
]

[dependency-groups]
dev = [
    "ipykernel>=7.2.0",
    "pytest>=8.0.0",
]
```

**Step 2: Install dependencies**

Run: `uv sync`
Expected: all packages install successfully.

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add torch, lightning, hydra, xgboost dependencies"
```

---

## Task 2: QWK Metric and Threshold Optimizer

**Files:**
- Create: `src/metrics/__init__.py` (empty)
- Create: `src/metrics/qwk.py`
- Create: `tests/metrics/__init__.py` (empty)
- Create: `tests/metrics/test_qwk.py`

**Step 1: Write the failing tests**

File: `tests/metrics/test_qwk.py`

```python
import numpy as np
import pytest
from src.metrics.qwk import quadratic_weighted_kappa, optimize_thresholds


class TestQWK:
    def test_perfect_agreement(self):
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        assert quadratic_weighted_kappa(y, y) == pytest.approx(1.0)

    def test_no_agreement(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([8, 8, 8, 8])
        kappa = quadratic_weighted_kappa(y_true, y_pred)
        assert kappa < 0.0

    def test_partial_agreement(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 4])
        kappa = quadratic_weighted_kappa(y_true, y_pred)
        assert 0.5 < kappa < 1.0

    def test_symmetric(self):
        y_true = np.array([1, 3, 5, 7])
        y_pred = np.array([2, 3, 4, 8])
        assert quadratic_weighted_kappa(y_true, y_pred) == pytest.approx(
            quadratic_weighted_kappa(y_pred, y_true)
        )


class TestThresholdOptimizer:
    def test_perfect_continuous(self):
        """Continuous predictions exactly at class centers should yield high QWK."""
        y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8] * 10)
        y_cont = y_true.astype(float)
        thresholds, kappa = optimize_thresholds(y_true, y_cont)
        assert len(thresholds) == 7
        assert kappa > 0.95

    def test_noisy_predictions(self):
        """Noisy but correlated predictions should still produce reasonable QWK."""
        rng = np.random.RandomState(42)
        y_true = rng.randint(1, 9, size=200)
        y_cont = y_true + rng.normal(0, 0.5, size=200)
        thresholds, kappa = optimize_thresholds(y_true, y_cont)
        assert kappa > 0.5

    def test_thresholds_are_sorted(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(1, 9, size=200)
        y_cont = y_true + rng.normal(0, 1.0, size=200)
        thresholds, _ = optimize_thresholds(y_true, y_cont)
        assert all(thresholds[i] <= thresholds[i + 1] for i in range(len(thresholds) - 1))
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/metrics/test_qwk.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.metrics'`

**Step 3: Implement QWK and threshold optimizer**

File: `src/metrics/qwk.py`

```python
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Quadratic Weighted Kappa between two ordinal rating arrays."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def _apply_thresholds(y_cont: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Map continuous predictions to ordinal classes 1-8 using 7 thresholds."""
    return np.digitize(y_cont, thresholds) + 1


def optimize_thresholds(
    y_true: np.ndarray,
    y_cont: np.ndarray,
    num_classes: int = 8,
) -> tuple[np.ndarray, float]:
    """Find optimal rounding thresholds that maximize QWK.

    Args:
        y_true: Ground truth ordinal labels (1 to num_classes).
        y_cont: Continuous model predictions.
        num_classes: Number of ordinal classes.

    Returns:
        Tuple of (sorted thresholds array of length num_classes-1, best QWK score).
    """
    # Initialize thresholds at class boundaries: 1.5, 2.5, ..., 7.5
    initial = np.arange(1.5, num_classes, 1.0)

    def neg_qwk(thresholds):
        t = np.sort(thresholds)
        preds = _apply_thresholds(y_cont, t)
        preds = np.clip(preds, 1, num_classes)
        return -quadratic_weighted_kappa(y_true, preds)

    result = minimize(neg_qwk, initial, method="Nelder-Mead",
                      options={"maxiter": 1000, "xatol": 1e-4})
    best_thresholds = np.sort(result.x)
    best_kappa = -result.fun
    return best_thresholds, best_kappa
```

File: `src/metrics/__init__.py`

```python
```

File: `tests/metrics/__init__.py`

```python
```

**Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/metrics/test_qwk.py -v`
Expected: all 7 tests PASS.

**Step 5: Commit**

```bash
git add src/metrics/ tests/metrics/
git commit -m "feat: add QWK metric and threshold optimizer with tests"
```

---

## Task 3: KAN Layer Implementations (ChebyKAN + FourierKAN)

**Files:**
- Create: `src/models/__init__.py` (empty)
- Create: `src/models/kan_layers.py`
- Create: `tests/models/__init__.py` (empty)
- Create: `tests/models/test_kan_layers.py`

**Step 1: Write failing tests**

File: `tests/models/test_kan_layers.py`

```python
import torch
import pytest
from src.models.kan_layers import ChebyKANLayer, FourierKANLayer


class TestChebyKANLayer:
    def test_output_shape(self):
        layer = ChebyKANLayer(in_features=10, out_features=5, degree=3)
        x = torch.randn(32, 10)
        out = layer(x)
        assert out.shape == (32, 5)

    def test_degree_one(self):
        layer = ChebyKANLayer(in_features=4, out_features=3, degree=1)
        x = torch.randn(8, 4)
        out = layer(x)
        assert out.shape == (8, 3)

    def test_gradient_flows(self):
        layer = ChebyKANLayer(in_features=6, out_features=2, degree=4)
        x = torch.randn(4, 6, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert all(p.grad is not None for p in layer.parameters())

    def test_input_clamping(self):
        """Layer should handle inputs outside [-1, 1] via tanh normalization."""
        layer = ChebyKANLayer(in_features=3, out_features=2, degree=3)
        x = torch.tensor([[10.0, -10.0, 5.0]])
        out = layer(x)
        assert torch.isfinite(out).all()


class TestFourierKANLayer:
    def test_output_shape(self):
        layer = FourierKANLayer(in_features=10, out_features=5, grid_size=4)
        x = torch.randn(32, 10)
        out = layer(x)
        assert out.shape == (32, 5)

    def test_grid_size_one(self):
        layer = FourierKANLayer(in_features=4, out_features=3, grid_size=1)
        x = torch.randn(8, 4)
        out = layer(x)
        assert out.shape == (8, 3)

    def test_gradient_flows(self):
        layer = FourierKANLayer(in_features=6, out_features=2, grid_size=5)
        x = torch.randn(4, 6, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert all(p.grad is not None for p in layer.parameters())

    def test_handles_large_input(self):
        """Fourier layer should handle any input range."""
        layer = FourierKANLayer(in_features=3, out_features=2, grid_size=4)
        x = torch.tensor([[100.0, -100.0, 50.0]])
        out = layer(x)
        assert torch.isfinite(out).all()
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/models/test_kan_layers.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement ChebyKAN and FourierKAN layers**

File: `src/models/kan_layers.py`

```python
import torch
import torch.nn as nn
import math


class ChebyKANLayer(nn.Module):
    """KAN layer using Chebyshev polynomial basis functions.

    Each edge (i, j) learns a function as a linear combination of
    Chebyshev polynomials T_0(x) through T_d(x). Inputs are normalized
    to [-1, 1] via tanh for numerical stability.
    """

    def __init__(self, in_features: int, out_features: int, degree: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        # Chebyshev coefficients: one per (output, input, polynomial degree)
        self.cheby_coeffs = nn.Parameter(
            torch.empty(out_features, in_features, degree + 1)
        )
        # Residual linear path for training stability
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.cheby_coeffs, mean=0.0,
                        std=1.0 / (self.in_features * (self.degree + 1)))
        nn.init.xavier_uniform_(self.base_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        # Normalize to [-1, 1] for Chebyshev domain
        x_norm = torch.tanh(x)

        # Build Chebyshev basis: T_0(x) = 1, T_1(x) = x, T_n(x) = 2x*T_{n-1} - T_{n-2}
        # Shape: (batch, in_features, degree+1)
        cheby = [torch.ones_like(x_norm), x_norm]
        for n in range(2, self.degree + 1):
            cheby.append(2 * x_norm * cheby[-1] - cheby[-2])
        cheby_basis = torch.stack(cheby, dim=-1)  # (batch, in, degree+1)

        # Compute KAN edge outputs: sum over basis weighted by learned coefficients
        # cheby_basis: (batch, in, degree+1)
        # cheby_coeffs: (out, in, degree+1)
        # Result: (batch, out)
        kan_out = torch.einsum("bid,oid->bo", cheby_basis, self.cheby_coeffs)

        # Add residual linear path
        base_out = nn.functional.linear(x, self.base_weight)

        return kan_out + base_out


class FourierKANLayer(nn.Module):
    """KAN layer using Fourier series basis functions.

    Each edge (i, j) learns a function as a sum of cosine and sine terms
    with learned coefficients. Supports any input range via learned
    frequency scaling.
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        # Fourier coefficients: cosine and sine for each frequency k=1..grid_size
        self.fourier_a = nn.Parameter(
            torch.empty(out_features, in_features, grid_size)
        )
        self.fourier_b = nn.Parameter(
            torch.empty(out_features, in_features, grid_size)
        )
        # Residual linear path
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.fourier_a, mean=0.0,
                        std=1.0 / (self.in_features * self.grid_size))
        nn.init.normal_(self.fourier_b, mean=0.0,
                        std=1.0 / (self.in_features * self.grid_size))
        nn.init.xavier_uniform_(self.base_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        # Frequency indices k = 1, 2, ..., grid_size
        k = torch.arange(1, self.grid_size + 1, device=x.device, dtype=x.dtype)

        # Scale input to [0, 2*pi] range using tanh -> [0, 1] -> [0, 2*pi]
        x_scaled = (torch.tanh(x) + 1) * math.pi  # (batch, in)

        # Build Fourier basis: cos(k*x) and sin(k*x)
        # x_scaled: (batch, in, 1) * k: (grid_size,) -> (batch, in, grid_size)
        x_k = x_scaled.unsqueeze(-1) * k  # (batch, in, grid_size)
        cos_basis = torch.cos(x_k)  # (batch, in, grid_size)
        sin_basis = torch.sin(x_k)  # (batch, in, grid_size)

        # Weighted sum over Fourier basis
        # fourier_a/b: (out, in, grid_size)
        kan_cos = torch.einsum("big,oig->bo", cos_basis, self.fourier_a)
        kan_sin = torch.einsum("big,oig->bo", sin_basis, self.fourier_b)

        # Residual linear path
        base_out = nn.functional.linear(x, self.base_weight)

        return kan_cos + kan_sin + base_out
```

File: `src/models/__init__.py`

```python
```

File: `tests/models/__init__.py`

```python
```

**Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/models/test_kan_layers.py -v`
Expected: all 8 tests PASS.

**Step 5: Commit**

```bash
git add src/models/ tests/models/
git commit -m "feat: implement ChebyKAN and FourierKAN layers with tests"
```

---

## Task 4: TabKAN Lightning Module

**Files:**
- Create: `src/models/tabkan.py`
- Create: `tests/models/test_tabkan.py`

**Step 1: Write failing tests**

File: `tests/models/test_tabkan.py`

```python
import torch
import pytest
from src.models.tabkan import TabKAN


class TestTabKAN:
    def test_chebykan_forward(self):
        model = TabKAN(
            in_features=20, widths=[32, 16], kan_type="chebykan", degree=3
        )
        x = torch.randn(8, 20)
        out = model(x)
        assert out.shape == (8, 1)

    def test_fourierkan_forward(self):
        model = TabKAN(
            in_features=20, widths=[32, 16], kan_type="fourierkan", grid_size=4
        )
        x = torch.randn(8, 20)
        out = model(x)
        assert out.shape == (8, 1)

    def test_single_layer(self):
        model = TabKAN(in_features=10, widths=[64], kan_type="chebykan", degree=2)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 1)

    def test_three_layers(self):
        model = TabKAN(
            in_features=50, widths=[128, 64, 32], kan_type="fourierkan", grid_size=5
        )
        x = torch.randn(4, 50)
        out = model(x)
        assert out.shape == (4, 1)

    def test_training_step(self):
        model = TabKAN(in_features=10, widths=[16], kan_type="chebykan", degree=3)
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        batch = (x, y)
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_validation_step(self):
        model = TabKAN(in_features=10, widths=[16], kan_type="chebykan", degree=3)
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        batch = (x, y)
        loss = model.validation_step(batch, 0)
        assert loss.ndim == 0

    def test_invalid_kan_type_raises(self):
        with pytest.raises(ValueError, match="Unknown kan_type"):
            TabKAN(in_features=10, widths=[16], kan_type="invalidkan")
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/models/test_tabkan.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement TabKAN**

File: `src/models/tabkan.py`

```python
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

        # Collect validation predictions for epoch-level QWK
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
        # Simple rounding QWK (threshold optimization is done post-training)
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
```

**Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/models/test_tabkan.py -v`
Expected: all 7 tests PASS.

**Step 5: Commit**

```bash
git add src/models/tabkan.py tests/models/test_tabkan.py
git commit -m "feat: implement TabKAN LightningModule with ChebyKAN/FourierKAN support"
```

---

## Task 5: MLP Baseline Lightning Module

**Files:**
- Create: `src/models/mlp.py`
- Create: `tests/models/test_mlp.py`

**Step 1: Write failing tests**

File: `tests/models/test_mlp.py`

```python
import torch
import pytest
from src.models.mlp import MLPBaseline


class TestMLPBaseline:
    def test_forward_shape(self):
        model = MLPBaseline(in_features=20, widths=[64, 32])
        x = torch.randn(8, 20)
        out = model(x)
        assert out.shape == (8, 1)

    def test_single_layer(self):
        model = MLPBaseline(in_features=10, widths=[32])
        x = torch.randn(4, 10)
        assert model(x).shape == (4, 1)

    def test_with_dropout(self):
        model = MLPBaseline(in_features=10, widths=[32, 16], dropout=0.5)
        model.eval()
        x = torch.randn(4, 10)
        assert model(x).shape == (4, 1)

    def test_training_step(self):
        model = MLPBaseline(in_features=10, widths=[16])
        batch = (torch.randn(4, 10), torch.randn(4, 1))
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/models/test_mlp.py -v`
Expected: FAIL

**Step 3: Implement MLP baseline**

File: `src/models/mlp.py`

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/models/test_mlp.py -v`
Expected: all 4 tests PASS.

**Step 5: Commit**

```bash
git add src/models/mlp.py tests/models/test_mlp.py
git commit -m "feat: add MLP baseline LightningModule"
```

---

## Task 6: XGBoost Baseline Wrapper

**Files:**
- Create: `src/models/xgb_baseline.py`
- Create: `tests/models/test_xgb_baseline.py`

**Step 1: Write failing tests**

File: `tests/models/test_xgb_baseline.py`

```python
import numpy as np
import pytest
from src.models.xgb_baseline import XGBBaseline


class TestXGBBaseline:
    def test_fit_predict(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = rng.randint(1, 9, size=100).astype(float)
        model = XGBBaseline(n_estimators=10, max_depth=3)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)

    def test_evaluate_returns_qwk(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = rng.randint(1, 9, size=100).astype(float)
        model = XGBBaseline(n_estimators=50, max_depth=3)
        model.fit(X, y)
        qwk = model.evaluate(X, y)
        assert -1.0 <= qwk <= 1.0
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/models/test_xgb_baseline.py -v`
Expected: FAIL

**Step 3: Implement XGBoost wrapper**

File: `src/models/xgb_baseline.py`

```python
import numpy as np
import xgboost as xgb
from src.metrics.qwk import optimize_thresholds, quadratic_weighted_kappa, _apply_thresholds


class XGBBaseline:
    """XGBoost regression baseline with QWK threshold optimization.

    Args:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        kwargs: Additional arguments passed to XGBRegressor.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs,
    ):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method="hist",
            random_state=42,
            **kwargs,
        )
        self.thresholds = None

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None):
        self.model.fit(X, y, eval_set=eval_set, verbose=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Predict, optimize thresholds, return QWK."""
        y_cont = self.predict(X)
        self.thresholds, kappa = optimize_thresholds(y_true, y_cont)
        return kappa

    def predict_ordinal(self, X: np.ndarray) -> np.ndarray:
        """Predict ordinal classes using optimized thresholds."""
        if self.thresholds is None:
            raise RuntimeError("Call evaluate() first to optimize thresholds.")
        y_cont = self.predict(X)
        return np.clip(_apply_thresholds(y_cont, self.thresholds), 1, 8)
```

**Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/models/test_xgb_baseline.py -v`
Expected: all 2 tests PASS.

**Step 5: Commit**

```bash
git add src/models/xgb_baseline.py tests/models/test_xgb_baseline.py
git commit -m "feat: add XGBoost baseline wrapper with QWK evaluation"
```

---

## Task 7: Lightning DataModule

**Files:**
- Create: `src/data/__init__.py` (empty)
- Create: `src/data/dataset.py`
- Create: `tests/data/__init__.py` (empty)
- Create: `tests/data/test_dataset.py`

**Context:** This wraps the existing `SOTAPreprocessor` from `src/data/sota_preprocessing.py`. It loads the Prudential CSV, preprocesses it, creates train/val splits, and builds missingness masks. The `SOTAPreprocessor.fit_transform` drops high-missingness features, encodes categoricals, imputes, and scales to [-1, 1].

**Step 1: Write failing tests**

File: `tests/data/test_dataset.py`

```python
import torch
import numpy as np
import pandas as pd
import pytest
from src.data.dataset import PrudentialDataModule


@pytest.fixture
def synthetic_data(tmp_path):
    """Create a minimal synthetic CSV mimicking Prudential structure."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "Id": range(n),
        "Product_Info_1": rng.choice([1, 2], n),
        "Product_Info_2": rng.choice(["A1", "B2", "C3"], n),
        "Product_Info_3": rng.randint(1, 30, n),
        "Product_Info_4": rng.uniform(0, 1, n),
        "Product_Info_5": rng.choice([1, 2], n),
        "Product_Info_6": rng.choice([1, 2, 3], n),
        "Product_Info_7": rng.choice([1, 2, 3], n),
        "Ins_Age": rng.uniform(0, 1, n),
        "Ht": rng.uniform(0, 1, n),
        "Wt": rng.uniform(0, 1, n),
        "BMI": rng.uniform(0.1, 0.9, n),
        "Employment_Info_1": rng.uniform(0, 0.1, n),
        "Employment_Info_2": rng.randint(1, 40, n),
        "Employment_Info_3": rng.choice([1, 3], n),
        "Employment_Info_4": rng.uniform(0, 1, n),
        "Employment_Info_5": rng.choice([1, 2, 3], n),
        "Employment_Info_6": rng.uniform(0, 1, n),
        "InsuredInfo_1": rng.randint(1, 4, n),
        "InsuredInfo_2": rng.choice([1, 2], n),
        "InsuredInfo_3": rng.randint(1, 12, n),
        "InsuredInfo_4": rng.choice([1, 2, 3], n),
        "InsuredInfo_5": rng.choice([1, 2], n),
        "InsuredInfo_6": rng.choice([1, 2], n),
        "InsuredInfo_7": rng.choice([1, 2, 3], n),
        "Insurance_History_1": rng.choice([1, 2], n),
        "Insurance_History_2": rng.choice([1, 2, 3], n),
        "Insurance_History_3": rng.choice([1, 2, 3], n),
        "Insurance_History_4": rng.choice([1, 2], n),
        "Insurance_History_5": rng.uniform(0, 1, n),
        "Insurance_History_7": rng.choice([1, 2, 3], n),
        "Insurance_History_8": rng.choice([1, 2, 3], n),
        "Insurance_History_9": rng.choice([1, 2, 3], n),
        "Family_Hist_1": rng.randint(1, 4, n),
        "Family_Hist_2": rng.uniform(0, 1, n),
        "Family_Hist_3": rng.uniform(0, 1, n),
        "Family_Hist_4": rng.uniform(0, 1, n),
        "Family_Hist_5": rng.uniform(0, 1, n),
        "Medical_History_1": rng.uniform(0, 200, n),
        "Response": rng.randint(1, 9, n),
    })
    # Add Medical_Keyword columns (binary)
    for i in range(1, 49):
        df[f"Medical_Keyword_{i}"] = rng.choice([0, 1], n, p=[0.9, 0.1])
    # Add some Medical_History columns with missing values
    for i in range(2, 42):
        if i not in [10, 15, 24, 32]:
            vals = rng.randint(0, 200, n).astype(float)
            vals[rng.random(n) < 0.3] = np.nan
            df[f"Medical_History_{i}"] = vals

    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path


class TestPrudentialDataModule:
    def test_setup_creates_splits(self, synthetic_data):
        dm = PrudentialDataModule(data_path=str(synthetic_data), val_split=0.2, batch_size=32)
        dm.setup()
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert len(dm.train_dataset) + len(dm.val_dataset) == 200

    def test_dataloaders_return_batches(self, synthetic_data):
        dm = PrudentialDataModule(data_path=str(synthetic_data), val_split=0.2, batch_size=16)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        x, y = batch
        assert x.ndim == 2
        assert y.ndim == 2
        assert y.shape[1] == 1

    def test_features_in_range(self, synthetic_data):
        dm = PrudentialDataModule(data_path=str(synthetic_data), val_split=0.2, batch_size=200)
        dm.setup()
        x, _ = next(iter(dm.train_dataloader()))
        assert x.min() >= -1.5  # Allow small tolerance from tanh normalization
        assert x.max() <= 1.5

    def test_num_features_property(self, synthetic_data):
        dm = PrudentialDataModule(data_path=str(synthetic_data), val_split=0.2, batch_size=32)
        dm.setup()
        assert dm.num_features > 0
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/data/test_dataset.py -v`
Expected: FAIL

**Step 3: Implement DataModule**

File: `src/data/dataset.py`

```python
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

    Args:
        data_path: Path to train.csv.
        val_split: Fraction of data for validation (default 0.2).
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers.
        use_sota: Whether to use SOTA preprocessing (True) or fallback.
        missing_threshold: Drop features with missingness above this rate.
        seed: Random seed for reproducibility.
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

        # Preprocess
        X_processed = self.preprocessor.fit_transform(X, y)
        X_np = X_processed.values.astype(np.float32)
        self._num_features = X_np.shape[1]

        # Stratified split
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=self.val_split, random_state=self.seed
        )
        train_idx, val_idx = next(sss.split(X_np, y))

        self.train_dataset = TabularDataset(X_np[train_idx], y[train_idx].astype(np.float32))
        self.val_dataset = TabularDataset(X_np[val_idx], y[val_idx].astype(np.float32))

        # Store numpy arrays for XGBoost baseline
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
```

File: `src/data/__init__.py`

```python
```

File: `tests/data/__init__.py`

```python
```

**Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/data/test_dataset.py -v`
Expected: all 4 tests PASS.

**Step 5: Commit**

```bash
git add src/data/__init__.py src/data/dataset.py tests/data/__init__.py tests/data/test_dataset.py
git commit -m "feat: add PrudentialDataModule with stratified splits"
```

---

## Task 8: Hydra Configuration Files

**Files:**
- Modify: `configs/config.yaml`
- Modify: `configs/model/tabkan.yaml` (rename to `configs/model/chebykan.yaml`)
- Create: `configs/model/fourierkan.yaml`
- Modify: `configs/model/mlp.yaml`
- Modify: `configs/model/xgb.yaml`
- Modify: `configs/train/default.yaml`
- Create: `configs/data/default.yaml`

**Step 1: Write all config files**

File: `configs/config.yaml`

```yaml
defaults:
  - model: chebykan
  - train: default
  - data: default
  - _self_

seed: 42
```

File: `configs/model/chebykan.yaml` (replace existing `tabkan.yaml`)

```yaml
name: chebykan
kan_type: chebykan
widths: [64, 32]
degree: 3
grid_size: null
lr: 1e-3
weight_decay: 1e-5
```

File: `configs/model/fourierkan.yaml`

```yaml
name: fourierkan
kan_type: fourierkan
widths: [64, 32]
degree: null
grid_size: 4
lr: 1e-3
weight_decay: 1e-5
```

File: `configs/model/mlp.yaml`

```yaml
name: mlp
widths: [128, 64]
dropout: 0.1
lr: 1e-3
weight_decay: 1e-5
```

File: `configs/model/xgb.yaml`

```yaml
name: xgb
n_estimators: 500
max_depth: 6
learning_rate: 0.1
```

File: `configs/train/default.yaml`

```yaml
max_epochs: 100
batch_size: 256
early_stopping_patience: 10
num_workers: 0
accelerator: auto
```

File: `configs/data/default.yaml`

```yaml
path: data/prudential-life-insurance-assessment/train.csv
val_split: 0.2
use_sota: true
missing_threshold: 0.5
```

**Step 2: Delete the old tabkan.yaml placeholder**

Run: `git rm configs/model/tabkan.yaml`

**Step 3: Commit**

```bash
git add configs/
git commit -m "feat: add Hydra config files for all models, training, and data"
```

---

## Task 9: Hydra Training Entry Point

**Files:**
- Modify: `main.py` (replace placeholder)
- Create: `src/train.py`
- Create: `src/evaluate.py`

**Step 1: Implement train.py**

File: `src/train.py`

```python
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from pathlib import Path
import numpy as np

from src.data.dataset import PrudentialDataModule
from src.models.tabkan import TabKAN
from src.models.mlp import MLPBaseline
from src.models.xgb_baseline import XGBBaseline
from src.metrics.qwk import optimize_thresholds


def build_model(cfg: DictConfig, num_features: int):
    """Instantiate the correct model from config."""
    if cfg.model.name == "xgb":
        return XGBBaseline(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            learning_rate=cfg.model.learning_rate,
        )
    elif cfg.model.name == "mlp":
        return MLPBaseline(
            in_features=num_features,
            widths=list(cfg.model.widths),
            dropout=cfg.model.dropout,
            lr=cfg.model.lr,
            weight_decay=cfg.model.weight_decay,
        )
    else:
        # chebykan or fourierkan
        return TabKAN(
            in_features=num_features,
            widths=list(cfg.model.widths),
            kan_type=cfg.model.kan_type,
            degree=cfg.model.get("degree", 3),
            grid_size=cfg.model.get("grid_size", 4),
            lr=cfg.model.lr,
            weight_decay=cfg.model.weight_decay,
        )


def train_xgb(cfg: DictConfig, dm: PrudentialDataModule):
    """Train and evaluate XGBoost baseline."""
    model = XGBBaseline(
        n_estimators=cfg.model.n_estimators,
        max_depth=cfg.model.max_depth,
        learning_rate=cfg.model.learning_rate,
    )
    model.fit(dm.X_train, dm.y_train,
              eval_set=[(dm.X_val, dm.y_val)])

    train_qwk = model.evaluate(dm.X_train, dm.y_train)
    val_qwk = model.evaluate(dm.X_val, dm.y_val)
    print(f"XGBoost — Train QWK: {train_qwk:.4f}, Val QWK: {val_qwk:.4f}")
    return model


def train_neural(cfg: DictConfig, dm: PrudentialDataModule):
    """Train a neural model (TabKAN or MLP) with Lightning."""
    model = build_model(cfg, dm.num_features)

    # Loggers
    loggers = []
    try:
        loggers.append(TensorBoardLogger("logs", name=cfg.model.name))
    except Exception:
        pass
    loggers.append(CSVLogger("logs", name=cfg.model.name))

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val/loss",
            patience=cfg.train.early_stopping_patience,
            mode="min",
        ),
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            filename=f"{cfg.model.name}-{{epoch}}-{{val/loss:.4f}}",
        ),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        callbacks=callbacks,
        logger=loggers,
        deterministic=True,
    )

    trainer.fit(model, dm)

    # Post-training threshold optimization
    model.eval()
    val_preds = []
    val_targets = []
    import torch
    with torch.no_grad():
        for batch in dm.val_dataloader():
            x, y = batch
            y_hat = model(x)
            val_preds.append(y_hat.cpu().numpy())
            val_targets.append(y.cpu().numpy())

    preds = np.concatenate(val_preds).flatten()
    targets = np.concatenate(val_targets).flatten()
    thresholds, final_qwk = optimize_thresholds(targets, preds)
    print(f"{cfg.model.name} — Final Val QWK (optimized thresholds): {final_qwk:.4f}")
    print(f"Thresholds: {thresholds}")

    return model, thresholds


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    L.seed_everything(cfg.seed)

    # Resolve data path relative to original working directory
    data_path = Path(hydra.utils.get_original_cwd()) / cfg.data.path

    dm = PrudentialDataModule(
        data_path=str(data_path),
        val_split=cfg.data.val_split,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        use_sota=cfg.data.use_sota,
        missing_threshold=cfg.data.missing_threshold,
        seed=cfg.seed,
    )
    dm.setup()

    if cfg.model.name == "xgb":
        train_xgb(cfg, dm)
    else:
        train_neural(cfg, dm)


if __name__ == "__main__":
    main()
```

File: `main.py` (update to point to train)

```python
"""Entry point — delegates to src.train.main()."""
from src.train import main

if __name__ == "__main__":
    main()
```

File: `src/evaluate.py`

```python
"""Standalone evaluation: load a checkpoint, optimize thresholds, report QWK."""
import argparse
import torch
import numpy as np
from pathlib import Path

from src.data.dataset import PrudentialDataModule
from src.models.tabkan import TabKAN
from src.models.mlp import MLPBaseline
from src.metrics.qwk import optimize_thresholds


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--model_type", type=str, choices=["chebykan", "fourierkan", "mlp"],
                        required=True)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    # Load data
    dm = PrudentialDataModule(data_path=args.data_path, val_split=args.val_split,
                              batch_size=args.batch_size)
    dm.setup()

    # Load model
    if args.model_type == "mlp":
        model = MLPBaseline.load_from_checkpoint(args.checkpoint)
    else:
        model = TabKAN.load_from_checkpoint(args.checkpoint)
    model.eval()

    # Predict on validation
    val_preds, val_targets = [], []
    with torch.no_grad():
        for x, y in dm.val_dataloader():
            val_preds.append(model(x).cpu().numpy())
            val_targets.append(y.cpu().numpy())

    preds = np.concatenate(val_preds).flatten()
    targets = np.concatenate(val_targets).flatten()

    thresholds, qwk = optimize_thresholds(targets, preds)
    print(f"Optimized QWK: {qwk:.4f}")
    print(f"Thresholds: {thresholds}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add src/train.py src/evaluate.py main.py
git commit -m "feat: add Hydra training entry point and evaluation script"
```

---

## Task 10: Add `__init__.py` Files and Verify Full Test Suite

**Files:**
- Create: `src/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)

**Step 1: Create init files**

Both files are empty `__init__.py` to make Python packages.

**Step 2: Run full test suite**

Run: `uv run python -m pytest tests/ -v`
Expected: all tests pass (QWK: 7, KAN layers: 8, TabKAN: 7, MLP: 4, XGBoost: 2, Dataset: 4 = ~32 tests).

**Step 3: Commit**

```bash
git add src/__init__.py tests/__init__.py
git commit -m "chore: add package init files"
```

---

## Task 11: Smoke Test — End-to-End Training Run

**Purpose:** Verify the full pipeline works end-to-end with real data. This is NOT a pytest — it's a manual integration check.

**Step 1: Run ChebyKAN training (short)**

Run: `uv run python src/train.py model=chebykan train.max_epochs=3`
Expected: trains for 3 epochs, prints loss and QWK, saves checkpoint.

**Step 2: Run FourierKAN training (short)**

Run: `uv run python src/train.py model=fourierkan train.max_epochs=3`
Expected: same as above.

**Step 3: Run MLP baseline (short)**

Run: `uv run python src/train.py model=mlp train.max_epochs=3`
Expected: same as above.

**Step 4: Run XGBoost baseline**

Run: `uv run python src/train.py model=xgb`
Expected: prints train and val QWK scores.

**Step 5: Commit any fixes needed, then final commit**

```bash
git add -A
git commit -m "chore: smoke test fixes and cleanup"
```

---

## Summary

| Task | What | Tests |
|------|------|-------|
| 1 | Dependencies | — |
| 2 | QWK + threshold optimizer | 7 |
| 3 | ChebyKAN + FourierKAN layers | 8 |
| 4 | TabKAN LightningModule | 7 |
| 5 | MLP baseline | 4 |
| 6 | XGBoost baseline | 2 |
| 7 | DataModule | 4 |
| 8 | Hydra configs | — |
| 9 | Train + evaluate entry points | — |
| 10 | Package init + full test suite | — |
| 11 | End-to-end smoke test | — |
