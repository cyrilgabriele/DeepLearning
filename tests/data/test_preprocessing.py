import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data import preprocess_kan_paper as kan_prep

def verify_kan_readiness(X_processed, feature_lists):
    """
    Analyzes processed data against KAN (Kolmogorov-Arnold Network) requirements.
    """
    print("\n" + "="*50)
    print("KAN READINESS ANALYSIS")
    print("="*50)
    
    # 1. Missingness Check
    missing_count = X_processed.isnull().sum().sum()
    print(f"Total Missing Values: {missing_count}")
    assert missing_count == 0, "KANs cannot handle missing values natively."
    
    # 2. Global Range Check
    # KAN splines are defined on a fixed grid (we enforce [-1, 1])
    min_val = X_processed.min().min()
    max_val = X_processed.max().max()
    print(f"Global Range: [{min_val:.4f}, {max_val:.4f}]")
    
    # 3. Distribution Analysis (Uniformity)
    # QuantileTransformer(output_distribution='uniform') should force features to be uniform.
    # This is critical for KANs to prevent spline activation saturation or grid-mismatch.
    print("\nFeature Distribution Summary (Top 10 features):")
    stats = X_processed.iloc[:, :10].describe().T[['min', '25%', '50%', '75%', 'max']]
    print(stats)
    
    # Check if mean is close to 0 and std is close to sqrt(1/3) ≈ 0.577 for uniform [-1, 1]
    means = X_processed.mean()
    stds = X_processed.std()

    print(f"\nAverage Mean across features: {means.mean():.4f} (Ideal: 0.0)")
    print(f"Average Std across features: {stds.mean():.4f} (Ideal: 0.577)")
    
    # 4. Feature Type Check
    all_numeric = all(np.issubdtype(dtype, np.number) for dtype in X_processed.dtypes)
    print(f"\nAll features numeric: {all_numeric}")
    
    # 5. Sensitivity to Spline Grids
    # We check if any feature has extreme outliers that QuantileTransformer might have missed
    # (Though QT is very robust to this).
    outliers_detected = (X_processed > 1.0).any().any() or (X_processed < -1.0).any().any()
    print(f"Out-of-bounds features detected (>1 or <0): {outliers_detected}")

    # 6. High Cardinality Encoding Check
    # CatBoost encoding should have turned Product_Info_2 into a single column.
    if "Product_Info_2" in X_processed.columns:
        print(f"Product_Info_2 successfully encoded. Sample value: {X_processed['Product_Info_2'].iloc[0]:.4f}")

    print("\nConclusion:")
    if missing_count == 0 and not outliers_detected and all_numeric:
        print("✅ Data is SOTA-ready for KAN architectures.")
        print("   - Uniform distribution optimizes spline grid coverage.")
        print("   - [-1, 1] range matches typical KAN initialization.")
        print("   - Low dimensionality (no one-hot expansion) prevents KAN parameter explosion.")
    else:
        print("❌ Data fails one or more KAN readiness checks.")

if __name__ == "__main__":
    # Correct path for local script execution (up 2 levels from tests/data/)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data" / "prudential-life-insurance-assessment"
    TRAIN_PATH = DATA_DIR / "train.csv"
    
    if TRAIN_PATH.exists():
        train = pd.read_csv(TRAIN_PATH)
        y = train["Response"]
        X = train.drop(columns=["Response"])
        
        print("Fitting preprocessing pipeline (this may take a minute)...")
        base_state = kan_prep.fit_preprocessor(train)
        X_base, _ = kan_prep.transform(train, base_state)
        kan_state = kan_prep.fit_kan_value_pipeline(X_base, base_state)
        X_processed, _ = kan_prep.transform(train, base_state, kan_state=kan_state)
        X_processed = pd.DataFrame(X_processed, columns=kan_state.feature_names)

        verify_kan_readiness(X_processed, kan_state.feature_names)
    else:
        print(f"Training data not found at {TRAIN_PATH}.")
