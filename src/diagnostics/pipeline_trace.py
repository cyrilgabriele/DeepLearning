"""Full pipeline trace — runs on synthetic data, logs every stage.

Usage:
    uv run python src/diagnostics/pipeline_trace.py

This script creates synthetic Prudential-like data and traces the entire
pipeline from raw input to ordinal predictions, printing diagnostics at
every stage. Use this to verify correctness without needing the real dataset.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import lightning as L

from src.data.prudential_features import get_feature_lists
from src.data.prudential_kan_preprocessing import PrudentialKANPreprocessor
from src.models.kan_layers import ChebyKANLayer, FourierKANLayer, BSplineKANLayer
from src.models.tabkan import TabKAN
from src.models.base import PrudentialModel
from src.models.mlp import MLPBaseline
from src.metrics.qwk import quadratic_weighted_kappa, optimize_thresholds, _apply_thresholds


def _header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _build_synthetic_prudential(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic DataFrame mimicking Prudential's column structure."""
    rng = np.random.RandomState(seed)

    data = {"Id": np.arange(1, n_samples + 1)}

    # Product_Info columns
    data["Product_Info_1"] = rng.choice([1, 2], n_samples)
    data["Product_Info_2"] = rng.choice(["A1", "B2", "C3", "D4", "E1"], n_samples)
    data["Product_Info_3"] = rng.randint(1, 40, n_samples)
    data["Product_Info_4"] = rng.uniform(0, 1, n_samples)
    data["Product_Info_5"] = rng.choice([1, 2], n_samples)
    data["Product_Info_6"] = rng.choice([1, 2, 3], n_samples)
    data["Product_Info_7"] = rng.choice([1, 2, 3], n_samples)

    # Continuous
    data["BMI"] = rng.normal(27, 5, n_samples).clip(10, 60)
    data["Ht"] = rng.normal(0.7, 0.1, n_samples).clip(0.4, 1.0)
    data["Wt"] = rng.normal(0.3, 0.1, n_samples).clip(0.05, 0.6)
    data["Ins_Age"] = rng.uniform(0, 1, n_samples)

    # Employment
    data["Employment_Info_1"] = rng.uniform(0, 0.1, n_samples)
    data["Employment_Info_2"] = rng.randint(1, 40, n_samples)
    data["Employment_Info_3"] = rng.choice([1, 3], n_samples)
    data["Employment_Info_4"] = rng.uniform(0, 1, n_samples)
    # Inject some NaN
    mask = rng.rand(n_samples) < 0.3
    data["Employment_Info_4"] = np.where(mask, np.nan, data["Employment_Info_4"])
    data["Employment_Info_5"] = rng.choice([1, 2, 3], n_samples)
    data["Employment_Info_6"] = rng.uniform(0, 1, n_samples)

    # InsuredInfo
    for i in range(1, 8):
        data[f"InsuredInfo_{i}"] = rng.randint(1, 4, n_samples)

    # Insurance_History
    data["Insurance_History_1"] = rng.choice([1, 2], n_samples)
    data["Insurance_History_2"] = rng.randint(1, 4, n_samples)
    data["Insurance_History_3"] = rng.randint(1, 4, n_samples)
    data["Insurance_History_4"] = rng.randint(1, 4, n_samples)
    data["Insurance_History_5"] = rng.uniform(0, 1, n_samples)
    data["Insurance_History_7"] = rng.randint(1, 4, n_samples)
    data["Insurance_History_8"] = rng.randint(1, 4, n_samples)
    data["Insurance_History_9"] = rng.randint(1, 4, n_samples)

    # Family_Hist
    data["Family_Hist_1"] = rng.randint(1, 4, n_samples)
    for i in [2, 3, 4, 5]:
        vals = rng.uniform(0, 1, n_samples)
        mask = rng.rand(n_samples) < 0.4
        data[f"Family_Hist_{i}"] = np.where(mask, np.nan, vals)

    # Medical_History (mix of categorical codes, continuous, binary)
    for i in range(1, 42):
        if i in [4, 22, 33, 38]:
            data[f"Medical_History_{i}"] = rng.choice([0, 1], n_samples)
        elif i in [1, 2, 10, 15, 24, 32]:
            vals = rng.uniform(0, 500, n_samples)
            mask = rng.rand(n_samples) < 0.5
            data[f"Medical_History_{i}"] = np.where(mask, np.nan, vals)
        else:
            data[f"Medical_History_{i}"] = rng.randint(0, 4, n_samples)

    # Medical_Keyword (binary)
    for i in range(1, 49):
        data[f"Medical_Keyword_{i}"] = rng.choice([0, 1], n_samples, p=[0.9, 0.1])

    # Target
    data["Response"] = rng.choice(range(1, 9), n_samples, p=[0.05, 0.1, 0.1, 0.1, 0.1, 0.15, 0.1, 0.3])

    return pd.DataFrame(data)


def stage_1_raw_data():
    """Stage 1: Build and inspect raw data."""
    _header("STAGE 1: Raw Data")
    df = _build_synthetic_prudential(500)
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)} ({len([c for c in df.columns if c not in ['Id', 'Response']])} features + Id + Response)")
    print(f"Target distribution:\n{df['Response'].value_counts().sort_index().to_string()}")
    print(f"Missing values (top 5):")
    missing = df.isnull().sum()
    for col in missing.nlargest(5).index:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]} ({missing[col]/len(df)*100:.1f}%)")
    return df


def stage_2_feature_classification(df):
    """Stage 2: Feature classification."""
    _header("STAGE 2: Feature Classification")
    X = df.drop(columns=["Response"])
    fl = get_feature_lists(X)
    for ftype, cols in fl.items():
        if ftype != "all":
            present = [c for c in cols if c in X.columns]
            print(f"  {ftype:12s}: {len(present):3d} features (of {len(cols)} defined)")
    # Check for overlaps
    cat = set(fl["categorical"])
    binary = set(fl["binary"])
    cont = set(fl["continuous"])
    overlap_cb = cat & binary
    overlap_cc = cat & cont
    if overlap_cb:
        print(f"  WARNING: Overlap categorical & binary: {overlap_cb}")
    if overlap_cc:
        print(f"  WARNING: Overlap categorical & continuous: {overlap_cc}")
    print(f"  Unassigned: {len(fl['ordinal'])} features classified as ordinal (residual)")
    return fl


def stage_3_preprocessing(df):
    """Stage 3: Full preprocessing pipeline."""
    _header("STAGE 3: Preprocessing (PrudentialKANPreprocessor)")
    y = df["Response"]
    X = df.drop(columns=["Response"])

    preprocessor = PrudentialKANPreprocessor(missing_threshold=0.5)
    X_processed = preprocessor.fit_transform(X, y)

    print(f"Input shape:  {X.shape}")
    print(f"Output shape: {X_processed.shape}")
    print(f"Dropped features: {preprocessor.dropped_features}")
    print(f"Missing indicator columns added: {len([c for c in X_processed.columns if c.startswith('missing_')])}")
    print(f"\nValue ranges:")
    print(f"  Global min: {X_processed.min().min():.4f}")
    print(f"  Global max: {X_processed.max().max():.4f}")
    print(f"  Mean of means: {X_processed.mean().mean():.4f}")
    print(f"  NaN count: {X_processed.isnull().sum().sum()}")
    print(f"  Dtypes: {X_processed.dtypes.unique()}")

    # Check per-type ranges
    fl = preprocessor.feature_lists
    for ftype in ["binary", "continuous", "ordinal", "categorical"]:
        cols = [c for c in fl[ftype] if c in X_processed.columns]
        if cols:
            subset = X_processed[cols]
            print(f"  {ftype:12s}: min={subset.min().min():.4f}, max={subset.max().max():.4f}, "
                  f"mean={subset.mean().mean():.4f}, cols={len(cols)}")

    return X_processed, y, preprocessor


def stage_4_model_forward(X_processed, y):
    """Stage 4: Forward pass through each model type."""
    _header("STAGE 4: Model Forward Pass (all architectures)")

    X_np = X_processed.values.astype(np.float32)
    in_features = X_np.shape[1]
    batch = torch.tensor(X_np[:32])  # Small batch of 32

    results = {}
    for kan_type, kwargs in [
        ("chebykan", {"degree": 3}),
        ("fourierkan", {"grid_size": 4}),
        ("bsplinekan", {"grid_size": 5, "spline_order": 3}),
    ]:
        model = TabKAN(in_features=in_features, widths=[64, 32], kan_type=kan_type, **kwargs)
        model.eval()
        with torch.no_grad():
            out = model(batch)

        out_np = out.numpy().flatten()
        results[kan_type] = out_np
        print(f"\n  {kan_type}:")
        print(f"    Input:  shape={batch.shape}, range=[{batch.min():.4f}, {batch.max():.4f}]")
        print(f"    Output: shape={out.shape}, range=[{out_np.min():.4f}, {out_np.max():.4f}]")
        print(f"    Mean={out_np.mean():.4f}, Std={out_np.std():.4f}")
        print(f"    Params: {sum(p.numel() for p in model.parameters()):,}")

        # Check for NaN/Inf
        if np.any(np.isnan(out_np)) or np.any(np.isinf(out_np)):
            print(f"    *** WARNING: NaN or Inf in output! ***")
        else:
            print(f"    No NaN/Inf: OK")

    # MLP baseline
    from src.models.mlp import MLPBaseline
    mlp = MLPBaseline(in_features=in_features, widths=[128, 64])
    mlp.eval()
    with torch.no_grad():
        mlp_out = mlp(batch).numpy().flatten()
    print(f"\n  MLP baseline:")
    print(f"    Output: shape={(32, 1)}, range=[{mlp_out.min():.4f}, {mlp_out.max():.4f}]")
    print(f"    Params: {sum(p.numel() for p in mlp.parameters()):,}")

    return results


def stage_5_kan_layer_internals(X_processed):
    """Stage 5: Trace through a single KAN layer to show basis function values."""
    _header("STAGE 5: KAN Layer Internals (ChebyKAN example)")

    x = torch.tensor(X_processed.values[:4].astype(np.float32))  # 4 samples
    in_f = x.shape[1]

    layer = ChebyKANLayer(in_f, 8, degree=3)

    # Manual trace
    x_norm = torch.tanh(x)
    print(f"  Input x:      shape={x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")
    print(f"  After tanh:   shape={x_norm.shape}, range=[{x_norm.min():.4f}, {x_norm.max():.4f}]")

    cheby = [torch.ones_like(x_norm), x_norm]
    for n in range(2, 4):
        cheby.append(2 * x_norm * cheby[-1] - cheby[-2])
    cheby_basis = torch.stack(cheby, dim=-1)
    print(f"  Cheby basis:  shape={cheby_basis.shape} (batch, in_features, degree+1)")
    print(f"    T_0 range: [{cheby_basis[:,:,0].min():.4f}, {cheby_basis[:,:,0].max():.4f}] (always 1)")
    print(f"    T_1 range: [{cheby_basis[:,:,1].min():.4f}, {cheby_basis[:,:,1].max():.4f}]")
    print(f"    T_2 range: [{cheby_basis[:,:,2].min():.4f}, {cheby_basis[:,:,2].max():.4f}]")
    print(f"    T_3 range: [{cheby_basis[:,:,3].min():.4f}, {cheby_basis[:,:,3].max():.4f}]")

    with torch.no_grad():
        out = layer(x)
    print(f"  Layer output: shape={out.shape}, range=[{out.min():.4f}, {out.max():.4f}]")


def stage_6_training_loop(X_processed, y):
    """Stage 6: Short training loop to verify loss decreases."""
    _header("STAGE 6: Training Loop (3 epochs, ChebyKAN)")

    L.seed_everything(42)
    X_np = X_processed.values.astype(np.float32)
    y_np = y.values.astype(np.float32)

    model = TabKAN(in_features=X_np.shape[1], widths=[32, 16], kan_type="chebykan", degree=3, lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_t = torch.tensor(X_np)
    y_t = torch.tensor(y_np).unsqueeze(1)

    losses = []
    model.train()
    for epoch in range(3):
        optimizer.zero_grad()
        y_hat = model(X_t)
        loss = model.loss_fn(y_hat, y_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  Epoch {epoch}: loss={loss.item():.4f}")

    if losses[-1] < losses[0]:
        print(f"  Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f} -- OK")
    else:
        print(f"  *** WARNING: Loss did NOT decrease: {losses[0]:.4f} -> {losses[-1]:.4f} ***")

    return model, X_np, y_np


def stage_7_threshold_optimization(model, X_np, y_np):
    """Stage 7: Continuous output -> threshold optimization -> ordinal."""
    _header("STAGE 7: Output -> Threshold Optimization -> Ordinal Classes")

    model.eval()
    with torch.no_grad():
        preds_cont = model(torch.tensor(X_np)).numpy().flatten()

    print(f"  Continuous predictions: shape={preds_cont.shape}")
    print(f"    Range: [{preds_cont.min():.4f}, {preds_cont.max():.4f}]")
    print(f"    Mean:  {preds_cont.mean():.4f}")
    print(f"    Std:   {preds_cont.std():.4f}")

    # Naive rounding
    naive = np.clip(np.round(preds_cont), 1, 8).astype(int)
    naive_qwk = quadratic_weighted_kappa(y_np.astype(int), naive)
    print(f"\n  Naive rounding QWK: {naive_qwk:.4f}")
    print(f"  Naive class distribution: {dict(zip(*np.unique(naive, return_counts=True)))}")

    # Optimized thresholds
    thresholds, opt_qwk = optimize_thresholds(y_np.astype(int), preds_cont)
    ordinal = np.clip(_apply_thresholds(preds_cont, thresholds), 1, 8)
    print(f"\n  Optimized thresholds: {np.round(thresholds, 3)}")
    print(f"  Optimized QWK: {opt_qwk:.4f} (improvement: {opt_qwk - naive_qwk:+.4f})")
    print(f"  Ordinal class distribution: {dict(zip(*np.unique(ordinal, return_counts=True)))}")

    # Validate
    unique_classes = np.unique(ordinal)
    assert all(1 <= c <= 8 for c in unique_classes), f"Invalid classes: {unique_classes}"
    print(f"  All classes in {{1..8}}: OK")
    assert len(thresholds) == 7, f"Expected 7 thresholds, got {len(thresholds)}"
    print(f"  Threshold count = 7: OK")
    assert np.all(np.diff(thresholds) > 0), "Thresholds not monotonically increasing!"
    print(f"  Thresholds monotonically increasing: OK")


def stage_8_registry_bridge():
    """Stage 8: Verify the registry bridge works (TabKANClassifier)."""
    _header("STAGE 8: Registry Bridge (TabKANClassifier via PrudentialModel)")

    from src.models.tabkan import build_tabkan_model

    model = build_tabkan_model("tabkan-tiny", random_state=42, flavor="chebykan")
    print(f"  Created: {type(model).__name__}")
    print(f"  Preset: tabkan-tiny, flavor: chebykan")
    print(f"  Widths: {model.widths}")
    print(f"  Implements PrudentialModel: {isinstance(model, PrudentialModel)}")

    # Build small synthetic data as DataFrame (what the Trainer provides)
    rng = np.random.RandomState(42)
    X_df = pd.DataFrame(rng.uniform(-1, 1, (100, 20)), columns=[f"f{i}" for i in range(20)])
    y_series = pd.Series(rng.choice(range(1, 9), 100))

    print(f"\n  Calling fit(X={X_df.shape}, y={y_series.shape})...")
    model.fit(X_df, y_series)
    print(f"  fit() completed. Module created: {model.module is not None}")

    preds = model.predict(X_df)
    print(f"  predict() output: shape={preds.shape}, dtype={preds.dtype}")
    print(f"  Prediction range: [{preds.min()}, {preds.max()}]")
    print(f"  Unique classes: {np.unique(preds)}")
    assert all(1 <= p <= 8 for p in preds), f"Invalid predictions: {preds}"
    print(f"  All predictions in {{1..8}}: OK")


def main():
    print("TabKAN Pipeline Trace — Full Diagnostic Run")
    print("=" * 70)

    df = stage_1_raw_data()
    stage_2_feature_classification(df)
    X_processed, y, preprocessor = stage_3_preprocessing(df)
    stage_4_model_forward(X_processed, y)
    stage_5_kan_layer_internals(X_processed)
    model, X_np, y_np = stage_6_training_loop(X_processed, y)
    stage_7_threshold_optimization(model, X_np, y_np)
    stage_8_registry_bridge()

    _header("ALL STAGES PASSED")
    print("  The pipeline is functional end-to-end.")
    print("  Preprocessing -> Model -> Output -> Thresholds -> Ordinal 1-8")
    print()


if __name__ == "__main__":
    main()
