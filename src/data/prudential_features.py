"""Feature taxonomy helpers for the Prudential Life Insurance dataset.

The dataset contains a mix of categorical codes, binary flags, and
continuous/ordinal measurements. We keep the logic centralized so the
paper-aligned and KAN-optimized preprocessing pipelines cannot drift.
"""

from typing import Dict, List
import pandas as pd


def get_feature_lists(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return feature groups inferred from Prudential metadata.

    Numerical values that behave categorically are grouped together so
    both preprocessors can apply consistent encoding and scaling.
    """

    all_features = [c for c in df.columns if c not in ["Id", "Response"]]

    categorical_features = [
        "Product_Info_2",
        "Product_Info_7",
        "InsuredInfo_1",
        "InsuredInfo_3",
        "Family_Hist_1",
        "Insurance_History_2",
        "Insurance_History_3",
        "Insurance_History_4",
        "Insurance_History_7",
        "Insurance_History_8",
        "Insurance_History_9",
    ]
    med_hist_codes = (
        [3]
        + list(range(5, 10))
        + list(range(11, 15))
        + list(range(16, 22))
        + [23]
        + list(range(25, 32))
        + list(range(34, 38))
        + list(range(39, 42))
    )
    categorical_features += [f"Medical_History_{i}" for i in med_hist_codes]

    binary_features = [c for c in all_features if "Medical_Keyword" in c]
    binary_features += [
        "Product_Info_1",
        "Product_Info_5",
        "Product_Info_6",
        "Employment_Info_3",
        "Employment_Info_5",
        "InsuredInfo_2",
        "InsuredInfo_4",
        "InsuredInfo_5",
        "InsuredInfo_6",
        "InsuredInfo_7",
        "Insurance_History_1",
        "Medical_History_4",
        "Medical_History_22",
        "Medical_History_33",
        "Medical_History_38",
    ]

    continuous_features = [
        "BMI",
        "Ht",
        "Wt",
        "Ins_Age",
        "Product_Info_4",
        "Employment_Info_1",
        "Employment_Info_4",
        "Employment_Info_6",
        "Family_Hist_2",
        "Family_Hist_3",
        "Family_Hist_4",
        "Family_Hist_5",
        "Insurance_History_5",
        "Medical_History_1",
        "Medical_History_2",
        "Medical_History_10",
        "Medical_History_15",
        "Medical_History_24",
        "Medical_History_32",
    ]

    all_assigned = set(categorical_features) | set(binary_features) | set(continuous_features)
    ordinal_features = [c for c in all_features if c not in all_assigned]

    return {
        "categorical": categorical_features,
        "binary": binary_features,
        "continuous": continuous_features,
        "ordinal": ordinal_features,
        "all": all_features,
    }

