import numpy as np
import pandas as pd
import torch
import pytest

from src.config import ExperimentConfig, ModelConfig, PreprocessingConfig, TrainerConfig
from src.models.tabkan import TabKAN, TabKANClassifier


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

    def test_bsplinekan_forward(self):
        model = TabKAN(
            in_features=20, widths=[32, 16], kan_type="bsplinekan", grid_size=5, spline_order=3
        )
        x = torch.randn(8, 20)
        out = model(x)
        assert out.shape == (8, 1)

    def test_invalid_kan_type_raises(self):
        with pytest.raises(ValueError, match="Unknown kan_type"):
            TabKAN(in_features=10, widths=[16], kan_type="invalidkan")

    def test_classifier_honours_lr_weight_decay_and_batch_size(self, monkeypatch):
        captured: dict[str, object] = {}

        class DummyTrainer:
            def __init__(self, **kwargs):
                captured["trainer_kwargs"] = kwargs

            def fit(self, module, **fit_kwargs):
                captured["module_hparams"] = dict(module.hparams)
                captured["train_batch_size"] = fit_kwargs["train_dataloaders"].batch_size
                val_loader = fit_kwargs.get("val_dataloaders")
                captured["val_batch_size"] = None if val_loader is None else val_loader.batch_size

        monkeypatch.setattr("src.models.tabkan.L.Trainer", DummyTrainer)

        model = TabKANClassifier(
            "tabkan-base",
            random_state=7,
            flavor="chebykan",
            hidden_widths=[16, 8],
            degree=4,
            max_epochs=3,
            lr=0.003,
            weight_decay=0.0007,
            batch_size=19,
            sparsity_lambda=0.02,
            l1_weight=1.5,
            entropy_weight=0.4,
        )

        rng = np.random.RandomState(7)
        X = pd.DataFrame(rng.randn(40, 6), columns=[f"f{i}" for i in range(6)])
        y = pd.Series(rng.randint(1, 9, size=40))

        model.fit(X, y, validation_data=(X.iloc[:10], y.iloc[:10]))

        assert captured["module_hparams"]["lr"] == pytest.approx(0.003)
        assert captured["module_hparams"]["weight_decay"] == pytest.approx(0.0007)
        assert captured["train_batch_size"] == 19
        assert captured["val_batch_size"] == 19

    def test_experiment_config_rejects_missing_tabkan_lr(self, tmp_path):
        with pytest.raises(ValueError, match="missing required training parameter\\(s\\): lr"):
            ExperimentConfig(
                trainer=TrainerConfig(
                    experiment_name="tabkan-missing-lr",
                    train_csv=tmp_path / "train.csv",
                    test_csv=None,
                    seed=42,
                ),
                preprocessing=PreprocessingConfig(recipe="kan_paper"),
                model=ModelConfig(
                    name="tabkan-base",
                    flavor="chebykan",
                    hidden_widths=[32, 16],
                    degree=3,
                    params={
                        "max_epochs": 5,
                        "weight_decay": 1e-5,
                        "batch_size": 32,
                        "sparsity_lambda": 0.0,
                        "l1_weight": 1.0,
                        "entropy_weight": 1.0,
                    },
                ),
            )
