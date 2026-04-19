from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_CANDIDATES = [
    ROOT_DIR / "data" / "Loan_default.csv",
    ROOT_DIR / "Loan_default.csv",
]
TARGET_COL = "Default"
ID_COL = "LoanID"


def print_section(title: str) -> None:
    line = "=" * 88
    print(f"\n{line}\n{title}\n{line}")


def locate_dataset() -> Path:
    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = "\n".join(f"- {p}" for p in DATASET_CANDIDATES)
    raise FileNotFoundError(f"Dataset not found. Checked:\n{searched}")


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    processed_dir = ROOT_DIR / "processed_data"
    output_dir = ROOT_DIR / "outputs" / "week3"
    model_dir = ROOT_DIR / "models"

    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print_section("WEEK 3 - DATA PREPROCESSING PIPELINE")

    dataset_path = locate_dataset()
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully")
    print(f"Source path: {dataset_path}")
    print(f"Initial shape: {df.shape}")

    print_section("COLUMN NAMES")
    for idx, col_name in enumerate(df.columns, start=1):
        print(f"{idx:02d}. {col_name}")

    print_section("MISSING VALUES SUMMARY")
    missing_summary = df.isna().sum().sort_values(ascending=False)
    print(missing_summary.to_string())
    print(f"Total missing values: {int(missing_summary.sum())}")

    if TARGET_COL in df.columns:
        print_section("TARGET DISTRIBUTION")
        target_counts = df[TARGET_COL].value_counts().sort_index()
        total_rows = len(df)
        for cls, count in target_counts.items():
            ratio = (count / total_rows) * 100
            print(f"Class {cls}: {count} rows ({ratio:.2f}%)")
    else:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    working_df = df.copy()
    if ID_COL in working_df.columns:
        working_df = working_df.drop(columns=[ID_COL])
        print(f"\nRemoved identifier column: {ID_COL}")

    categorical_cols = working_df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    if TARGET_COL in categorical_cols:
        categorical_cols.remove(TARGET_COL)

    numerical_cols = working_df.select_dtypes(include=["number"]).columns.tolist()
    if TARGET_COL in numerical_cols:
        numerical_cols.remove(TARGET_COL)

    print_section("FEATURE TYPE SPLIT")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

    print_section("MISSING VALUE HANDLING")
    fill_report: dict[str, str] = {}
    for col in numerical_cols:
        if working_df[col].isna().any():
            fill_value = working_df[col].median()
            working_df[col] = working_df[col].fillna(fill_value)
            fill_report[col] = f"median={fill_value}"
    for col in categorical_cols:
        if working_df[col].isna().any():
            mode_series = working_df[col].mode(dropna=True)
            fill_value = mode_series.iloc[0] if not mode_series.empty else "UNKNOWN"
            working_df[col] = working_df[col].fillna(fill_value)
            fill_report[col] = f"mode={fill_value}"

    if fill_report:
        print("Applied missing value filling:")
        for col, method in fill_report.items():
            print(f"- {col}: {method}")
    else:
        print("No missing values detected. Filling step skipped.")

    print_section("ENCODING + SCALING")
    encoded_df = pd.get_dummies(
        working_df,
        columns=categorical_cols,
        drop_first=True,
        dtype=int,
    )
    print("Categorical columns encoded")

    scaler = StandardScaler()
    if numerical_cols:
        scaled_values = scaler.fit_transform(encoded_df[numerical_cols])
        scaled_numeric_df = pd.DataFrame(
            scaled_values,
            columns=numerical_cols,
            index=encoded_df.index,
        )
        encoded_df[numerical_cols] = scaled_numeric_df
        print("Numerical columns scaled")
    else:
        print("No numerical columns found; scaling skipped")

    print(f"Processed dataset shape: {encoded_df.shape}")

    processed_csv_path = processed_dir / "week3_processed_dataset.csv"
    scaler_path = model_dir / "week3_scaler.pkl"
    column_list_path = output_dir / "week3_processed_columns.txt"
    summary_json_path = output_dir / "week3_preprocessing_summary.json"
    summary_txt_path = output_dir / "week3_preprocessing_summary.txt"

    encoded_df.to_csv(processed_csv_path, index=False)
    joblib.dump(scaler, scaler_path)

    save_text(column_list_path, "\n".join(encoded_df.columns.tolist()))

    summary_payload = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "initial_shape": list(df.shape),
        "processed_shape": list(encoded_df.shape),
        "target_distribution": df[TARGET_COL].value_counts().sort_index().to_dict(),
        "categorical_columns": categorical_cols,
        "numerical_columns": numerical_cols,
        "total_missing_values_initial": int(df.isna().sum().sum()),
        "processed_dataset_path": str(processed_csv_path),
        "scaler_path": str(scaler_path),
        "columns_path": str(column_list_path),
    }
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_txt = (
        "Week 3 Preprocessing Summary\n"
        f"- Timestamp: {summary_payload['run_timestamp']}\n"
        f"- Dataset: {summary_payload['dataset_path']}\n"
        f"- Initial shape: {summary_payload['initial_shape']}\n"
        f"- Processed shape: {summary_payload['processed_shape']}\n"
        f"- Target distribution: {summary_payload['target_distribution']}\n"
        f"- Categorical columns: {len(categorical_cols)}\n"
        f"- Numerical columns: {len(numerical_cols)}\n"
        f"- Missing values (initial): {summary_payload['total_missing_values_initial']}\n"
        f"- Processed CSV: {summary_payload['processed_dataset_path']}\n"
        f"- Scaler file: {summary_payload['scaler_path']}\n"
        f"- Encoded columns list: {summary_payload['columns_path']}\n"
    )
    save_text(summary_txt_path, summary_txt)

    print_section("OUTPUTS SAVED")
    print(f"Processed dataset: {processed_csv_path}")
    print(f"Scaler artifact: {scaler_path}")
    print(f"Columns list: {column_list_path}")
    print(f"Run summary (json): {summary_json_path}")
    print(f"Run summary (txt): {summary_txt_path}")

    print_section("WEEK 3 COMPLETED")
    print("Preprocessing pipeline finished successfully.")


if __name__ == "__main__":
    main()
