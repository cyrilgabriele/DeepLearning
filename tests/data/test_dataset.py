import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from configs import set_global_seed
from src.preprocessing.dataset import PrudentialDataModule


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
    for i in range(1, 49):
        df[f"Medical_Keyword_{i}"] = rng.choice([0, 1], n, p=[0.9, 0.1])
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
        seed = set_global_seed(777)
        dm = PrudentialDataModule(data_path=str(synthetic_data), batch_size=32, seed=seed)
        dm.setup()
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert len(dm.train_dataset) + len(dm.val_dataset) == 200

    def test_dataloaders_return_batches(self, synthetic_data):
        seed = set_global_seed(778)
        dm = PrudentialDataModule(data_path=str(synthetic_data), batch_size=16, seed=seed)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        x, y = batch
        assert x.ndim == 2
        assert y.ndim == 2
        assert y.shape[1] == 1

    def test_features_are_finite(self, synthetic_data):
        seed = set_global_seed(779)
        dm = PrudentialDataModule(data_path=str(synthetic_data), batch_size=200, seed=seed)
        dm.setup()
        x, _ = next(iter(dm.train_dataloader()))
        assert torch.isfinite(x).all()

    def test_num_features_property(self, synthetic_data):
        seed = set_global_seed(780)
        dm = PrudentialDataModule(data_path=str(synthetic_data), batch_size=32, seed=seed)
        dm.setup()
        assert dm.num_features > 0
