import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from category_encoders import CatBoostEncoder, TargetEncoder
from pathlib import Path

def get_feature_lists(df):
    """
    Categorizes features based on the Prudential dataset structure.
    """
    all_features = [c for c in df.columns if c not in ["Id", "Response"]]
    
    categorical_features = ["Product_Info_2"]
    
    binary_features = [c for c in all_features if "Medical_Keyword" in c]
    binary_features += ["InsuredInfo_2", "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7",
                       "Employment_Info_3", "Employment_Info_5", "Insurance_History_1",
                       "Medical_History_22", "Medical_History_33", "Medical_History_38", "Medical_History_4",
                       "Product_Info_1", "Product_Info_5", "Product_Info_6"]
    
    continuous_features = ["BMI", "Ht", "Wt", "Ins_Age", "Employment_Info_1", "Employment_Info_4", 
                           "Employment_Info_6", "Insurance_History_5", "Medical_History_1", 
                           "Product_Info_4"]
    continuous_features += [c for c in all_features if "Family_Hist" in c and c not in binary_features]
    
    ordinal_features = [c for c in all_features if c not in categorical_features + binary_features + continuous_features]
    
    return {
        "categorical": categorical_features,
        "binary": binary_features,
        "continuous": continuous_features,
        "ordinal": ordinal_features,
        "all": all_features
    }

class SOTAPreprocessor:
    def __init__(self, missing_threshold=0.5, use_sota=True):
        self.missing_threshold = missing_threshold
        self.use_sota = use_sota
        self.feature_lists = None
        self.dropped_features = []
        
        if use_sota:
            self.imputer = IterativeImputer(random_state=42, max_iter=10)
            self.encoder = CatBoostEncoder()
            # Scaling is CRITICAL for KAN. We apply it to ALL features to ensure [0, 1] range.
            self.scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
        else:
            self.imputer = SimpleImputer(strategy='median', add_indicator=True)
            self.encoder = TargetEncoder()
            self.scaler = MinMaxScaler()

    def fit_transform(self, X, y):
        X = X.copy()
        
        # 1. Drop high missingness features
        missing_rate = X.isnull().mean()
        self.dropped_features = missing_rate[missing_rate > self.missing_threshold].index.tolist()
        X = X.drop(columns=self.dropped_features)
        
        self.feature_lists = get_feature_lists(X)
        
        # 2. Categorical Encoding
        if "Product_Info_2" in X.columns:
            X["Product_Info_2"] = self.encoder.fit_transform(X["Product_Info_2"], y)
        
        # 3. Imputation
        X_no_id = X.drop(columns=["Id"], errors='ignore')
        X_imputed = self.imputer.fit_transform(X_no_id)
        
        if not self.use_sota:
            indicator_names = [f"missing_{X_no_id.columns[i]}" for i in self.imputer.indicator_.features_]
            new_cols = list(X_no_id.columns) + indicator_names
        else:
            new_cols = X_no_id.columns
            
        X = pd.DataFrame(X_imputed, columns=new_cols, index=X.index)
        
        # 4. Scaling (SOTA: QuantileTransformation for KAN)
        # For KANs, we scale ALL processed columns to [0, 1].
        # Even binary features should be consistently mapped, though QT on binary might be noisy, 
        # it ensures the strict [0, 1] requirement for spline grids.
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        return X

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.dropped_features, errors='ignore')
        
        if "Product_Info_2" in X.columns:
            X["Product_Info_2"] = self.encoder.transform(X["Product_Info_2"])
            
        X_no_id = X.drop(columns=["Id"], errors='ignore')
        X_imputed = self.imputer.transform(X_no_id)
        
        if not self.use_sota:
            indicator_names = [f"missing_{X_no_id.columns[i]}" for i in self.imputer.indicator_.features_]
            new_cols = list(X_no_id.columns) + indicator_names
        else:
            new_cols = X_no_id.columns
            
        X = pd.DataFrame(X_imputed, columns=new_cols, index=X.index)
        X = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        
        return X

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data" / "prudential-life-insurance-assessment"
    TRAIN_PATH = DATA_DIR / "train.csv"
    
    if TRAIN_PATH.exists():
        print(f"Loading data from {TRAIN_PATH}...")
        train = pd.read_csv(TRAIN_PATH)
        y = train["Response"]
        X = train.drop(columns=["Response"])
        
        print("Running SOTA Preprocessing optimized for KAN...")
        preprocessor = SOTAPreprocessor(use_sota=True)
        X_processed = preprocessor.fit_transform(X, y)
        
        print(f"Original shape: {X.shape}")
        print(f"Processed shape: {X_processed.shape}")
        print(f"Global Min: {X_processed.min().min():.4f}, Global Max: {X_processed.max().max():.4f}")
        print(f"Missing values: {X_processed.isnull().sum().sum()}")
    else:
        print(f"Data file not found at {TRAIN_PATH}.")
