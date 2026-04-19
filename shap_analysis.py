from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import DMatrix

from preprocessing import (
    DATA_PATH,
    ROOT_DIR,
    TARGET_COL,
    apply_feature_engineering,
    encode_training_features,
    load_dataset,
    scale_to_frame,
)


def build_test_frame(
    feature_names: list[str],
    scaler,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = load_dataset(DATA_PATH)
    df = apply_feature_engineering(df)
    y = df[TARGET_COL].astype(int)
    x = encode_training_features(df.drop(columns=[TARGET_COL]))
    x = x.reindex(columns=feature_names, fill_value=0)

    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    x_test_scaled = scale_to_frame(scaler, x_test, feature_names)
    return x_test, x_test_scaled, y_test


def assign_error_types(y_true: pd.Series, y_pred: np.ndarray) -> pd.Series:
    labels = pd.Series("Unknown", index=y_true.index)
    labels.loc[(y_true == 1) & (y_pred == 1)] = "True_Positive"
    labels.loc[(y_true == 0) & (y_pred == 0)] = "True_Negative"
    labels.loc[(y_true == 0) & (y_pred == 1)] = "False_Positive"
    labels.loc[(y_true == 1) & (y_pred == 0)] = "False_Negative"
    return labels


def save_group_analysis(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    error_types: pd.Series,
    output_dir: Path,
) -> dict:
    analysis_df = x_test.copy()
    analysis_df["TrueLabel"] = y_test
    analysis_df["PredictedLabel"] = y_pred
    analysis_df["DefaultProbability"] = y_proba
    analysis_df["ErrorType"] = error_types

    means_path = output_dir / "error_analysis_feature_means.csv"
    counts_path = output_dir / "error_analysis_counts.csv"
    matrix_path = output_dir / "error_analysis_confusion_matrix.csv"

    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    analysis_df.groupby("ErrorType")[numeric_cols].mean().round(4).to_csv(means_path)
    analysis_df["ErrorType"].value_counts().rename_axis("ErrorType").reset_index(name="count").to_csv(
        counts_path,
        index=False,
    )

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(matrix_path)

    return {
        "means_path": str(means_path),
        "counts_path": str(counts_path),
        "confusion_matrix_path": str(matrix_path),
        "counts": analysis_df["ErrorType"].value_counts().to_dict(),
    }


def shap_summary_plot(model, x_group: pd.DataFrame, output_path: Path, title: str) -> str | None:
    if x_group.empty:
        return None

    sample = x_group.sample(n=min(500, len(x_group)), random_state=42)

    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        plt.figure()
        shap.summary_plot(shap_values, sample, show=False)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return str(output_path)
    except Exception as exc:
        dmatrix = DMatrix(sample, feature_names=sample.columns.tolist())
        contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
        contrib_values = contribs[:, :-1]
        importance = pd.Series(
            np.abs(contrib_values).mean(axis=0),
            index=sample.columns,
        ).sort_values(ascending=False).head(20)

        plt.figure(figsize=(8, 6))
        importance.sort_values().plot(kind="barh", color="#2563eb")
        plt.title(f"{title} (XGBoost contribution fallback)")
        plt.xlabel("Mean absolute contribution")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"SHAP library was unavailable or failed for {title}; fallback plot saved. Reason: {exc}")
        return str(output_path)


def main() -> None:
    output_dir = ROOT_DIR / "outputs" / "shap"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(ROOT_DIR / "xgboost_optimized.pkl")
    scaler = joblib.load(ROOT_DIR / "scaler.pkl")
    feature_names = joblib.load(ROOT_DIR / "feature_names.pkl")
    threshold_path = ROOT_DIR / "decision_threshold.pkl"
    decision_threshold = joblib.load(threshold_path) if threshold_path.exists() else 0.50

    x_test, x_test_scaled, y_test = build_test_frame(feature_names, scaler)

    y_proba = model.predict_proba(x_test_scaled)[:, 1]
    y_pred = (y_proba >= decision_threshold).astype(int)
    error_types = assign_error_types(y_test, y_pred)

    group_analysis = save_group_analysis(
        x_test,
        y_test,
        y_pred,
        y_proba,
        error_types,
        output_dir,
    )

    fp_idx = error_types[error_types == "False_Positive"].index
    fn_idx = error_types[error_types == "False_Negative"].index

    fp_plot = shap_summary_plot(
        model,
        x_test_scaled.loc[fp_idx],
        output_dir / "shap_summary_False_Positives.png",
        "SHAP Summary - False Positives",
    )
    fn_plot = shap_summary_plot(
        model,
        x_test_scaled.loc[fn_idx],
        output_dir / "shap_summary_False_Negatives.png",
        "SHAP Summary - False Negatives",
    )

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "decision_threshold": decision_threshold,
        "group_analysis": group_analysis,
        "false_positive_shap_plot": fp_plot,
        "false_negative_shap_plot": fn_plot,
    }
    summary_path = output_dir / "shap_error_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("SHAP and error analysis complete.")
    print(f"False Positive plot: {fp_plot}")
    print(f"False Negative plot: {fn_plot}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
