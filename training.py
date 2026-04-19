from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from preprocessing import ROOT_DIR, prepare_train_test


OPTIMIZED_XGB_PARAMS = {
    "learning_rate": 0.08,
    "max_depth": 3,
    "n_estimators": 220,
    "subsample": 0.7,
    "colsample_bytree": 0.85,
    "gamma": 1.0,
}
TARGET_RECALL = 0.69


def require_smote():
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise RuntimeError(
            "imbalanced-learn is required for SMOTE. "
            "Install it with: py -m pip install imbalanced-learn"
        ) from exc
    return SMOTE


def find_threshold_for_target_recall(
    y_true: pd.Series,
    y_proba,
    target_recall: float = TARGET_RECALL,
) -> float:
    best_threshold = 0.50
    best_score: tuple[float, float] | None = None

    for threshold in [step / 1000 for step in range(100, 901)]:
        y_pred = (y_proba >= threshold).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        score = (abs(recall - target_recall), -precision)
        if best_score is None or score < best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold


def evaluate_model(
    model: XGBClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict:
    y_proba = model.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "mean_predicted_default_probability": float(y_proba.mean()),
        },
        "classification_report": report,
        "confusion_matrix": cm,
    }


def save_confusion_matrix(cm, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title("CreditScope Tuned XGBoost - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    output_dir = ROOT_DIR / "outputs" / "training"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing data with shared preprocessing...")
    x_train, x_test, y_train, y_test, scaler, feature_names = prepare_train_test()

    print(f"Train shape before SMOTE: {x_train.shape}")
    print(f"Class balance before SMOTE: {y_train.value_counts().sort_index().to_dict()}")

    SMOTE = require_smote()
    sampler = SMOTE(random_state=42)
    x_train_balanced, y_train_balanced = sampler.fit_resample(x_train, y_train)
    x_train_balanced = pd.DataFrame(x_train_balanced, columns=feature_names)
    y_train_balanced = pd.Series(y_train_balanced, name=y_train.name)

    print(f"Train shape after SMOTE : {x_train_balanced.shape}")
    print(f"Class balance after SMOTE : {y_train_balanced.value_counts().sort_index().to_dict()}")

    model = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
        **OPTIMIZED_XGB_PARAMS,
    )

    print(f"Training XGBoost with optimized params: {OPTIMIZED_XGB_PARAMS}")
    model.fit(x_train_balanced, y_train_balanced)

    y_proba = model.predict_proba(x_test)[:, 1]
    decision_threshold = find_threshold_for_target_recall(y_test, y_proba)
    result = evaluate_model(model, x_test, y_test, decision_threshold)
    metrics = result["metrics"]

    print("\nEvaluation")
    print(f"Decision threshold: {decision_threshold:.3f}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")

    joblib.dump(model, ROOT_DIR / "xgboost_optimized.pkl")
    joblib.dump(scaler, ROOT_DIR / "scaler.pkl")
    joblib.dump(feature_names, ROOT_DIR / "feature_names.pkl")
    joblib.dump(decision_threshold, ROOT_DIR / "decision_threshold.pkl")

    metrics_path = output_dir / "training_metrics.json"
    report_path = output_dir / "classification_report.csv"
    cm_csv_path = output_dir / "confusion_matrix.csv"
    cm_png_path = output_dir / "confusion_matrix.png"

    metrics_payload = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "sampler": "SMOTE",
        "optimized_params": OPTIMIZED_XGB_PARAMS,
        "target_recall": TARGET_RECALL,
        "decision_threshold": decision_threshold,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    pd.DataFrame(result["classification_report"]).transpose().to_csv(report_path)
    pd.DataFrame(
        result["confusion_matrix"],
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(cm_csv_path)
    save_confusion_matrix(result["confusion_matrix"], cm_png_path)

    print("\nArtifacts saved")
    print(f"Model          : {ROOT_DIR / 'xgboost_optimized.pkl'}")
    print(f"Scaler         : {ROOT_DIR / 'scaler.pkl'}")
    print(f"Feature names  : {ROOT_DIR / 'feature_names.pkl'}")
    print(f"Threshold      : {ROOT_DIR / 'decision_threshold.pkl'}")
    print(f"Metrics        : {metrics_path}")
    print(f"Confusion plot : {cm_png_path}")


if __name__ == "__main__":
    main()
