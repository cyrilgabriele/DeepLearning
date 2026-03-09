import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from category_encoders import CatBoostEncoder, TargetEncoder
from pathlib import Path

def get_feature_lists(df):
    """
    Categorizes features based on the Prudential dataset structure and provided insights.
    Numerical values that represent categories (nominal) are moved to categorical.
    Binary flags are moved to binary and handled with simple linear scaling.
    Continuous/Ordinal features are handled with Quantile/MinMax Transformation.
    """
    all_features = [c for c in df.columns if c not in ["Id", "Response"]]
    
    # 1. Categorical (Nominal Codes / Strings)
    categorical_features = [
        "Product_Info_2", "Product_Info_7", 
        "InsuredInfo_1", "InsuredInfo_3", 
        "Family_Hist_1", 
        "Insurance_History_2", "Insurance_History_3", "Insurance_History_4", 
        "Insurance_History_7", "Insurance_History_8", "Insurance_History_9"
    ]
    med_hist_codes = [3] + list(range(5, 10)) + list(range(11, 15)) + list(range(16, 22)) + [23] + list(range(25, 32)) + list(range(34, 38)) + list(range(39, 42))
    categorical_features += [f"Medical_History_{i}" for i in med_hist_codes]
    
    # 2. Binary (Flags 0/1 or similar)
    binary_features = [c for c in all_features if "Medical_Keyword" in c]
    binary_features += [
        "Product_Info_1", "Product_Info_5", "Product_Info_6", 
        "Employment_Info_3", "Employment_Info_5", 
        "InsuredInfo_2", "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7", 
        "Insurance_History_1", 
        "Medical_History_4", "Medical_History_22", "Medical_History_33", "Medical_History_38"
    ]
    
    # 3. Continuous
    # High-cardinality measured features
    continuous_features = [
        "BMI", "Ht", "Wt", "Ins_Age", 
        "Product_Info_4", 
        "Employment_Info_1", "Employment_Info_4", "Employment_Info_6", 
        "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5", 
        "Insurance_History_5", 
        "Medical_History_1", "Medical_History_2", "Medical_History_10", "Medical_History_15", "Medical_History_24", "Medical_History_32"
    ]
    
    # 4. Ordinal / Discrete (Remaining)
    # Includes low-cardinality 'continuous' features (Product_Info_3, Employment_Info_2)
    # for smoother spline learning in KANs.
    all_assigned = set(categorical_features) | set(binary_features) | set(continuous_features)
    ordinal_features = [c for c in all_features if c not in all_assigned]
    
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
        self.binary_params = {} # Stores min/max for consistency
        
        # IterativeImputer with add_indicator captures important missingness signal
        if use_sota:
            self.imputer = IterativeImputer(random_state=42, max_iter=10, add_indicator=True)
            self.encoder = CatBoostEncoder()
            self.scaler_cont = QuantileTransformer(output_distribution='uniform', random_state=42)
            self.scaler_ord = MinMaxScaler(feature_range=(-1, 1))
            self.scaler_cat = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.imputer = SimpleImputer(strategy='median', add_indicator=True)
            self.encoder = TargetEncoder()
            self.scaler_cont = MinMaxScaler(feature_range=(-1, 1))
            self.scaler_ord = MinMaxScaler(feature_range=(-1, 1))
            self.scaler_cat = MinMaxScaler(feature_range=(-1, 1))

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
        
        # Determine names for original and new indicator columns
        orig_cols = list(X_no_id.columns)
        indicator_names = [f"missing_{orig_cols[i]}" for i in self.imputer.indicator_.features_]
        new_cols = orig_cols + indicator_names
        
        X = pd.DataFrame(X_imputed, columns=new_cols, index=X.index)
        
        # Update feature lists to include new binary indicators
        self.feature_lists["binary"] = list(set(self.feature_lists["binary"]) | set(indicator_names))
        
        # 4. Specialized Scaling for KANs
        X_final = X.copy()
        
        # A. Binary: Store parameters for consistent transform
        binary_cols = [c for c in self.feature_lists["binary"] if c in X.columns]
        for col in binary_cols:
            c_min, c_max = X[col].min(), X[col].max()
            self.binary_params[col] = (c_min, c_max)
            if c_max > c_min:
                X_final[col] = 2 * (X[col] - c_min) / (c_max - c_min) - 1
            else:
                X_final[col] = 0.0

        # B. Continuous
        cont_cols = [c for c in self.feature_lists["continuous"] if c in X.columns]
        if cont_cols:
            scaled = self.scaler_cont.fit_transform(X[cont_cols])
            if self.use_sota:
                scaled = scaled * 2 - 1
            X_final[cont_cols] = scaled

        # C. Ordinal
        ord_cols = [c for c in self.feature_lists["ordinal"] if c in X.columns]
        if ord_cols:
            X_final[ord_cols] = self.scaler_ord.fit_transform(X[ord_cols])

        # D. Categorical
        cat_cols = [c for c in self.feature_lists["categorical"] if c in X.columns]
        if cat_cols:
            X_final[cat_cols] = self.scaler_cat.fit_transform(X[cat_cols])
            
        # 5. Explicit Clipping for KAN Spline Stability
        return X_final.clip(-1, 1)

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.dropped_features, errors='ignore')
        
        # 2. Categorical Encoding
        cols_to_encode = [c for c in self.feature_lists["categorical"] if c in X.columns]
        if cols_to_encode:
            X[cols_to_encode] = self.encoder.transform(X[cols_to_encode])
            
        X_no_id = X.drop(columns=["Id"], errors='ignore')
        X_imputed = self.imputer.transform(X_no_id)
        
        orig_cols = list(X_no_id.columns)
        indicator_names = [f"missing_{orig_cols[i]}" for i in self.imputer.indicator_.features_]
        new_cols = orig_cols + indicator_names
        
        X = pd.DataFrame(X_imputed, columns=new_cols, index=X.index)
        X_final = X.copy()
        
        # A. Binary: Use stored parameters
        binary_cols = [c for c in self.feature_lists["binary"] if c in X.columns]
        for col in binary_cols:
            if col in self.binary_params:
                c_min, c_max = self.binary_params[col]
                if c_max > c_min:
                    X_final[col] = 2 * (X[col] - c_min) / (c_max - c_min) - 1
                else:
                    X_final[col] = 0.0
            else:
                X_final[col] = 0.0

        # B. Continuous
        cont_cols = [c for c in self.feature_lists["continuous"] if c in X.columns]
        if cont_cols:
            scaled = self.scaler_cont.transform(X[cont_cols])
            if self.use_sota:
                scaled = scaled * 2 - 1
            X_final[cont_cols] = scaled

        # C. Ordinal
        ord_cols = [c for c in self.feature_lists["ordinal"] if c in X.columns]
        if ord_cols:
            X_final[ord_cols] = self.scaler_ord.transform(X[ord_cols])

        # D. Categorical
        cat_cols = [c for c in self.feature_lists["categorical"] if c in X.columns]
        if cat_cols:
            X_final[cat_cols] = self.scaler_cat.transform(X[cat_cols])
            
        return X_final.clip(-1, 1)

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
