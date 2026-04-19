from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "Loan_default.csv"
TARGET_COL = "Default"

CATEGORICAL_COLUMNS = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]


def _numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same engineered features for training and API inference."""
    df = df.copy()

    income = _numeric_column(df, "Income")
    loan_amount = _numeric_column(df, "LoanAmount")
    age = _numeric_column(df, "Age")

    safe_income = income.replace(0, np.nan)
    dti_ratio = (loan_amount / safe_income).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Wiki alignment: derive DTI from raw income and requested loan amount.
    df["DTIRatio"] = dti_ratio
    df["Age_Income_Interaction"] = age * income

    return df


def load_dataset(filepath: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if "LoanID" in df.columns:
        df = df.drop(columns=["LoanID"])
    return df


def _present_columns(columns: Iterable[str], df: pd.DataFrame) -> list[str]:
    return [column for column in columns if column in df.columns]


def encode_training_features(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = _present_columns(CATEGORICAL_COLUMNS, df)
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)


def encode_inference_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    categorical_cols = _present_columns(CATEGORICAL_COLUMNS, df)
    encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)
    return encoded.reindex(columns=feature_names, fill_value=0)


def scale_to_frame(
    scaler: StandardScaler,
    x: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    columns = feature_names or x.columns.tolist()
    scaled = scaler.transform(x)
    return pd.DataFrame(scaled, columns=columns, index=x.index)


def prepare_train_test(
    filepath: str | Path = DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler, list[str]]:
    df = load_dataset(filepath)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' was not found in {filepath}")

    df = apply_feature_engineering(df)
    y = df[TARGET_COL].astype(int)
    x_raw = df.drop(columns=[TARGET_COL])
    x = encode_training_features(x_raw)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    feature_names = x.columns.tolist()
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=feature_names,
        index=x_train.index,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=feature_names,
        index=x_test.index,
    )

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler, feature_names


def prepare_inference_frame(
    raw_application: dict,
    feature_names: list[str],
    scaler: StandardScaler,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_df = pd.DataFrame([raw_application])
    engineered_df = apply_feature_engineering(raw_df)
    encoded_df = encode_inference_features(engineered_df, feature_names)
    scaled_df = scale_to_frame(scaler, encoded_df, feature_names)
    return engineered_df, encoded_df, scaled_df


def main() -> None:
    print("Loading and preprocessing dataset...")
    x_train, x_test, y_train, y_test, _, feature_names = prepare_train_test()
    print(f"Training shape : {x_train.shape}")
    print(f"Test shape     : {x_test.shape}")
    print(f"Target balance : {y_train.value_counts().sort_index().to_dict()}")
    print(f"Feature count  : {len(feature_names)}")
    print("Engineered features: DTIRatio, Age_Income_Interaction")


if __name__ == "__main__":
    main()
