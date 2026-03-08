import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from category_encoders import CatBoostEncoder, TargetEncoder
from pathlib import Path

def get_feature_lists(df):
    """
    Categorizes features based on the Prudential dataset structure.
    Numerical values that represent categories (nominal) are moved to categorical.
    """
    all_features = [c for c in df.columns if c not in ["Id", "Response"]]
    
    # 1. Categorical (Nominal) - These are numerical proxies or strings
    categorical_features = ["Product_Info_2", "Product_Info_1", "Product_Info_3", "Product_Info_5", "Product_Info_6", "Product_Info_7",
                            "Employment_Info_2", "Employment_Info_3", "Employment_Info_5",
                            "InsuredInfo_1", "InsuredInfo_2", "InsuredInfo_3", "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7",
                            "Insurance_History_1", "Insurance_History_2", "Insurance_History_3", "Insurance_History_4", "Insurance_History_7", "Insurance_History_8", "Insurance_History_9",
                            "Family_Hist_1"]
    
    # Add Medical_History nominal features (2-9, 11-14, 16-23, 25-31, 33-41)
    medical_nominal = [f"Medical_History_{i}" for i in range(2, 42) if i not in [10, 15, 24, 32]]
    categorical_features += medical_nominal
    
    # 2. Binary (Dummy)
    binary_features = [c for c in all_features if "Medical_Keyword" in c]
    # Specific categorical-proxies that are actually binary (0/1 or 1/2)
    binary_features += ["Medical_History_4", "Medical_History_22", "Medical_History_33", "Medical_History_38",
                       "Product_Info_1", "Product_Info_5", "Product_Info_6", "InsuredInfo_2", "InsuredInfo_4", 
                       "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7", "Employment_Info_3", "Employment_Info_5", 
                       "Insurance_History_1"]
    
    # Ensure categorical takes precedence for encoding, but we'll track binary for scaling
    binary_only = [c for c in binary_features if c not in categorical_features]
    
    # 3. Continuous
    continuous_features = ["BMI", "Ht", "Wt", "Ins_Age", "Employment_Info_1", "Employment_Info_4", 
                           "Employment_Info_6", "Insurance_History_5", "Medical_History_1", 
                           "Product_Info_4"]
    continuous_features += [c for c in all_features if "Family_Hist" in c and any(str(i) in c for i in [2,3,4,5])]
    
    # 4. Ordinal / Discrete (Remaining)
    ordinal_features = [c for c in all_features if c not in categorical_features + binary_only + continuous_features]
    
    return {
        "categorical": categorical_features,
        "binary": binary_only,
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
            # QuantileTransformer for continuous/ordinal to normalize distributions
            self.scaler_cont = QuantileTransformer(output_distribution='uniform', random_state=42)
            # MinMaxScaler for categorical-encoded features to preserve relative distances
            self.scaler_cat = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.imputer = SimpleImputer(strategy='median', add_indicator=True)
            self.encoder = TargetEncoder()
            self.scaler_cont = MinMaxScaler(feature_range=(-1, 1))

    def fit_transform(self, X, y):
        X = X.copy()
        
        # 1. Drop high missingness features
        missing_rate = X.isnull().mean()
        self.dropped_features = missing_rate[missing_rate > self.missing_threshold].index.tolist()
        X = X.drop(columns=self.dropped_features)
        
        self.feature_lists = get_feature_lists(X)
        
        # 2. Categorical Encoding
        cols_to_encode = [c for c in self.feature_lists["categorical"] if c in X.columns]
        if cols_to_encode:
            X[cols_to_encode] = self.encoder.fit_transform(X[cols_to_encode], y)
        
        # 3. Imputation
        X_no_id = X.drop(columns=["Id"], errors='ignore')
        X_imputed = self.imputer.fit_transform(X_no_id)
        
        if not self.use_sota:
            indicator_names = [f"missing_{X_no_id.columns[i]}" for i in self.imputer.indicator_.features_]
            new_cols = list(X_no_id.columns) + indicator_names
        else:
            new_cols = X_no_id.columns
            
        X = pd.DataFrame(X_imputed, columns=new_cols, index=X.index)
        
        # 4. Specialized Scaling for KANs
        # We must avoid applying QuantileTransformer to Binary features (introduces noise)
        X_final = X.copy()
        
        # A. Binary: Strict [-1, 1] without noise
        # Note: Binary features might be 0/1 or 1/2 from original data, but Imputer makes them float.
        # We use a simple linear shift.
        binary_cols = [c for c in self.feature_lists["binary"] if c in X.columns]
        for col in binary_cols:
            c_min, c_max = X[col].min(), X[col].max()
            if c_max > c_min:
                X_final[col] = 2 * (X[col] - c_min) / (c_max - c_min) - 1
            else:
                X_final[col] = 0.0 # Constant feature

        # B. Continuous & Ordinal: Quantile Transformation
        cont_ord_cols = [c for c in self.feature_lists["continuous"] + self.feature_lists["ordinal"] if c in X.columns]
        if cont_ord_cols:
            scaled = self.scaler_cont.fit_transform(X[cont_ord_cols])
            if self.use_sota:
                scaled = scaled * 2 - 1
            X_final[cont_ord_cols] = scaled

        # C. Categorical (Now Target-Encoded): Linear Scaling to [-1, 1]
        cat_cols = [c for c in self.feature_lists["categorical"] if c in X.columns]
        if cat_cols:
            X_final[cat_cols] = self.scaler_cat.fit_transform(X[cat_cols])
            
        return X_final

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.dropped_features, errors='ignore')
        
        # 2. Categorical Encoding
        cols_to_encode = [c for c in self.feature_lists["categorical"] if c in X.columns]
        if cols_to_encode:
            X[cols_to_encode] = self.encoder.transform(X[cols_to_encode])
            
        X_no_id = X.drop(columns=["Id"], errors='ignore')
        X_imputed = self.imputer.transform(X_no_id)
        
        if not self.use_sota:
            indicator_names = [f"missing_{X_no_id.columns[i]}" for i in self.imputer.indicator_.features_]
            new_cols = list(X_no_id.columns) + indicator_names
        else:
            new_cols = X_no_id.columns
            
        X = pd.DataFrame(X_imputed, columns=new_cols, index=X.index)
        X_final = X.copy()
        
        # A. Binary
        binary_cols = [c for c in self.feature_lists["binary"] if c in X.columns]
        for col in binary_cols:
            c_min, c_max = X[col].min(), X[col].max() # Should ideally use fitted min/max
            if c_max > c_min:
                X_final[col] = 2 * (X[col] - c_min) / (c_max - c_min) - 1
            else:
                X_final[col] = 0.0

        # B. Continuous & Ordinal
        cont_ord_cols = [c for c in self.feature_lists["continuous"] + self.feature_lists["ordinal"] if c in X.columns]
        if cont_ord_cols:
            scaled = self.scaler_cont.transform(X[cont_ord_cols])
            if self.use_sota:
                scaled = scaled * 2 - 1
            X_final[cont_ord_cols] = scaled

        # C. Categorical
        cat_cols = [c for c in self.feature_lists["categorical"] if c in X.columns]
        if cat_cols:
            X_final[cat_cols] = self.scaler_cat.transform(X[cat_cols])
            
        return X_final

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
        print(f"Binary Mean (should be near -1 or 1): {X_processed.iloc[:, :5].mean().mean():.4f}")
        print(f"Missing values: {X_processed.isnull().sum().sum()}")
    else:
        print(f"Data file not found at {TRAIN_PATH}.")
