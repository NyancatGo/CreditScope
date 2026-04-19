from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
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
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import DMatrix, XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET_COL = "Default"
DATASET_CANDIDATES = [
    ROOT_DIR / "processed_data" / "week4_feature_engineered_dataset.csv",
    ROOT_DIR / "processed_data" / "week3_processed_dataset.csv",
    ROOT_DIR / "week3_processed_dataset.csv",
]


def print_header() -> None:
    line = "=" * 96
    print(
        f"\n{line}\n"
        "CREDITSCOPE | HAFTA 6 | MODEL KARSILASTIRMA + TUNING + ILK SHAP\n"
        f"Calisma Zamani: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{line}"
    )


def print_block(tag: str, lines: list[str]) -> None:
    print(f"\n[{tag}]")
    for line in lines:
        print(f"- {line}")


def locate_dataset() -> Path:
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path
    checked = "\n".join(f"- {p}" for p in DATASET_CANDIDATES)
    raise FileNotFoundError(f"Hafta 6 veri seti bulunamadi. Kontrol edilen yollar:\n{checked}")


def apply_feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    created: list[str] = []

    if "Income_minus_LoanAmount" not in df.columns and {"Income", "LoanAmount"}.issubset(df.columns):
        df["Income_minus_LoanAmount"] = df["Income"] - df["LoanAmount"]
        created.append("Income_minus_LoanAmount")

    if "CreditScore_x_DTIRatio" not in df.columns and {"CreditScore", "DTIRatio"}.issubset(df.columns):
        df["CreditScore_x_DTIRatio"] = df["CreditScore"] * df["DTIRatio"]
        created.append("CreditScore_x_DTIRatio")

    if "Employment_per_CreditLine" not in df.columns and {"MonthsEmployed", "NumCreditLines"}.issubset(df.columns):
        df["Employment_per_CreditLine"] = df["MonthsEmployed"] / (df["NumCreditLines"] + 1.0)
        created.append("Employment_per_CreditLine")

    return df, created


def build_report_tr(report_dict: dict) -> str:
    label_map = {
        "0": "Sinif 0",
        "1": "Sinif 1",
        "macro avg": "Makro Ortalama",
        "weighted avg": "Agirlikli Ortalama",
        "accuracy": "Genel Dogruluk",
    }
    lines = []
    lines.append("Etiket                 Kesinlik    Duyarlilik    F1 Skoru     Destek")
    lines.append("-" * 72)

    for key in ["0", "1", "macro avg", "weighted avg"]:
        if key in report_dict:
            row = report_dict[key]
            lines.append(
                f"{label_map[key]:<22} {row['precision']:>9.4f} {row['recall']:>13.4f} "
                f"{row['f1-score']:>11.4f} {int(row['support']):>10d}"
            )

    if "accuracy" in report_dict:
        accuracy = float(report_dict["accuracy"])
        support = int(report_dict["weighted avg"]["support"])
        lines.append("-" * 72)
        lines.append(
            f"{label_map['accuracy']:<22} {'-':>9} {'-':>13} {accuracy:>11.4f} {support:>10d}"
        )
    return "\n".join(lines)


def evaluate(model: XGBClassifier, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = build_report_tr(report_dict)
    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "report_dict": report_dict,
        "report_text": report_text,
    }


def save_confusion_plot(cm, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def load_previous_model_results() -> pd.DataFrame:
    frames = []

    week4_path = ROOT_DIR / "outputs" / "week4" / "week4_baseline_metrics.csv"
    if week4_path.exists():
        df4 = pd.read_csv(week4_path)
        df4 = df4.rename(columns={"model": "model_name"})
        df4["source"] = "week4_baseline"
        frames.append(df4[["model_name", "accuracy", "precision", "recall", "f1", "source"]])

    week5_path = ROOT_DIR / "outputs" / "week5" / "week5_xgboost_metrics_comparison.csv"
    if week5_path.exists():
        df5 = pd.read_csv(week5_path)
        df5 = df5.rename(columns={"senaryo": "model_name"})
        df5["source"] = "week5_xgboost"
        frames.append(df5[["model_name", "accuracy", "precision", "recall", "f1", "source"]])

    if not frames:
        return pd.DataFrame(columns=["model_name", "accuracy", "precision", "recall", "f1", "source"])

    return pd.concat(frames, ignore_index=True)


def generate_shap_outputs(
    model: XGBClassifier,
    x_test: pd.DataFrame,
    shap_dir: Path,
) -> dict:
    shap_dir.mkdir(parents=True, exist_ok=True)
    sample_n = min(1200, len(x_test))
    sample_df = x_test.sample(n=sample_n, random_state=42)

    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_df)

        summary_dot_path = shap_dir / "week6_shap_summary_dot.png"
        summary_bar_path = shap_dir / "week6_shap_summary_bar.png"

        plt.figure()
        shap.summary_plot(shap_values, sample_df, show=False)
        plt.tight_layout()
        plt.savefig(summary_dot_path, dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, sample_df, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(summary_bar_path, dpi=150, bbox_inches="tight")
        plt.close()

        mean_abs = np.abs(shap_values).mean(axis=0)
        top_importance = (
            pd.Series(mean_abs, index=sample_df.columns).sort_values(ascending=False).head(20).reset_index()
        )
        top_importance.columns = ["feature", "mean_abs_shap"]
        top_csv = shap_dir / "week6_top_shap_features.csv"
        top_importance.to_csv(top_csv, index=False)

        return {
            "method": "shap_library",
            "primary_visual_path": str(summary_dot_path),
            "summary_bar_path": str(summary_bar_path),
            "top_features_csv": str(top_csv),
        }
    except Exception as exc:
        dmatrix = DMatrix(sample_df, feature_names=sample_df.columns.tolist())
        contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
        contrib_values = contribs[:, :-1]
        mean_abs = np.abs(contrib_values).mean(axis=0)

        importance = pd.Series(mean_abs, index=sample_df.columns).sort_values(ascending=False).head(20)
        shap_like_png = shap_dir / "week6_shap_benzeri_onem.png"
        shap_like_csv = shap_dir / "week6_shap_benzeri_onem.csv"

        plt.figure(figsize=(9, 6))
        importance.sort_values().plot(kind="barh")
        plt.title("XGBoost Katki Degerleri (SHAP Benzeri) - Top 20 Ozellik")
        plt.xlabel("Ortalama Mutlak Katki")
        plt.tight_layout()
        plt.savefig(shap_like_png, dpi=150)
        plt.close()

        importance.rename("mean_abs_contrib").to_csv(shap_like_csv, header=True)

        return {
            "method": "xgboost_pred_contribs",
            "primary_visual_path": str(shap_like_png),
            "top_features_csv": str(shap_like_csv),
            "fallback_reason": str(exc),
        }


def main() -> None:
    week6_dir = ROOT_DIR / "outputs" / "week6"
    shap_dir = ROOT_DIR / "outputs" / "shap"
    model_dir = ROOT_DIR / "models"
    week6_dir.mkdir(parents=True, exist_ok=True)
    shap_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print_header()

    previous_results = load_previous_model_results()
    comparison_path = week6_dir / "week6_previous_model_comparison.csv"
    if previous_results.empty:
        previous_summary = [
            "Onceki hafta metrik dosyalari bulunamadi.",
            f"Karsilastirma dosyasi (bos): {comparison_path}",
        ]
        pd.DataFrame(columns=["model_name", "accuracy", "precision", "recall", "f1", "source"]).to_csv(
            comparison_path, index=False
        )
    else:
        comparison_df = previous_results.sort_values(by=["recall", "f1"], ascending=False).reset_index(drop=True)
        comparison_df.to_csv(comparison_path, index=False)
        best_prev = comparison_df.iloc[0]
        top3 = ", ".join(
            f"{row.model_name} ({row.recall:.4f})"
            for row in comparison_df.head(3).itertuples(index=False)
        )
        previous_summary = [
            f"Onceki model kaydi: {len(comparison_df)}",
            f"En yuksek recall (onceki): {best_prev['model_name']} ({best_prev['recall']:.4f})",
            f"Top 3 recall: {top3}",
            f"Karsilastirma dosyasi: {comparison_path}",
        ]
    print_block("INFO", previous_summary)

    dataset_path = locate_dataset()
    df = pd.read_csv(dataset_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Hedef kolon '{TARGET_COL}' bulunamadi: {dataset_path}")

    df, engineered = apply_feature_engineering(df.copy())
    y = df[TARGET_COL]
    x = df.drop(columns=[TARGET_COL])
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    class_counts = y_train.value_counts().sort_index()
    scale_pos_weight = class_counts[0] / class_counts[1]

    print_block(
        "INFO",
        [
            f"Girdi veri seti: {dataset_path}",
            f"Girdi boyutu: {df.shape}",
            f"Egitim/Test boyutu: {x_train.shape} / {x_test.shape}",
            f"Ek ozellik sayisi: {len(engineered)}",
            f"scale_pos_weight: {scale_pos_weight:.4f}",
        ],
    )

    max_tuning_samples = 80000
    if len(x_train) > max_tuning_samples:
        x_tune, _, y_tune, _ = train_test_split(
            x_train,
            y_train,
            train_size=max_tuning_samples,
            stratify=y_train,
            random_state=42,
        )
    else:
        x_tune, y_tune = x_train, y_train

    base_model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        verbosity=0,
    )
    param_distributions = {
        "n_estimators": [150, 220, 300, 380],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08, 0.12],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.5, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="recall",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    print_block(
        "INFO",
        [
            "Hyperparameter tuning baslatildi (RandomizedSearchCV, recall odakli).",
            f"Tuning orneklem boyutu: {x_tune.shape}",
            "Arama ayari: n_iter=10, cv=3",
        ],
    )
    search.fit(x_tune, y_tune)

    best_params = search.best_params_
    best_cv_recall = float(search.best_score_)

    tuned_model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
        **best_params,
    )
    tuned_model.fit(x_train, y_train)
    tuned_result = evaluate(tuned_model, x_test, y_test)

    shap_outputs = generate_shap_outputs(tuned_model, x_test, shap_dir)

    metrics_df = pd.DataFrame(
        [
            {
                "model_name": "xgboost_tuned_recall",
                **tuned_result["metrics"],
            }
        ]
    )
    tuned_metrics_csv = week6_dir / "week6_tuned_xgboost_metrics.csv"
    tuned_metrics_txt = week6_dir / "week6_tuned_xgboost_metrics.txt"
    tuned_report_txt = week6_dir / "week6_tuned_xgboost_classification_report.txt"
    tuned_report_csv = week6_dir / "week6_tuned_xgboost_classification_report.csv"
    tuned_cm_csv = week6_dir / "week6_tuned_xgboost_confusion_matrix.csv"
    tuned_cm_png = week6_dir / "week6_tuned_xgboost_confusion_matrix.png"
    tuning_best_json = week6_dir / "week6_tuning_best_params.json"
    tuning_cv_top_csv = week6_dir / "week6_tuning_cv_top10.csv"
    run_summary_json = week6_dir / "week6_run_summary.json"
    tuned_model_path = model_dir / "week6_xgboost_tuned.pkl"

    metrics_df.to_csv(tuned_metrics_csv, index=False)
    tuned_metrics_txt.write_text(metrics_df.to_string(index=False), encoding="utf-8")
    tuned_report_txt.write_text(tuned_result["report_text"], encoding="utf-8")
    pd.DataFrame(tuned_result["report_dict"]).transpose().to_csv(tuned_report_csv)
    pd.DataFrame(
        tuned_result["confusion_matrix"],
        index=["gercek_0", "gercek_1"],
        columns=["tahmin_0", "tahmin_1"],
    ).to_csv(tuned_cm_csv)
    save_confusion_plot(tuned_result["confusion_matrix"], tuned_cm_png, "Week6 Tuned XGBoost - Karmasiklik Matrisi")

    cv_results = pd.DataFrame(search.cv_results_).sort_values("mean_test_score", ascending=False).head(10)
    cv_results[["mean_test_score", "std_test_score", "rank_test_score", "params"]].to_csv(
        tuning_cv_top_csv, index=False
    )

    tuning_best_json.write_text(
        json.dumps(
            {
                "best_params": best_params,
                "best_cv_recall": best_cv_recall,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    joblib.dump(tuned_model, tuned_model_path)

    run_summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_dataset": str(dataset_path),
        "input_shape": list(df.shape),
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "engineered_features": engineered,
        "scale_pos_weight": float(scale_pos_weight),
        "best_params": best_params,
        "best_cv_recall": best_cv_recall,
        "test_metrics": tuned_result["metrics"],
        "tuned_model_path": str(tuned_model_path),
        "shap_outputs": shap_outputs,
    }
    run_summary_json.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print_block(
        "SUMMARY",
        [
            f"Best parameters: {best_params}",
            f"Best CV recall: {best_cv_recall:.4f}",
            f"Test accuracy: {tuned_result['metrics']['accuracy']:.4f}",
            f"Test precision: {tuned_result['metrics']['precision']:.4f}",
            f"Test recall: {tuned_result['metrics']['recall']:.4f}",
            f"Test f1: {tuned_result['metrics']['f1']:.4f}",
            f"SHAP yontemi: {shap_outputs.get('method', 'bilinmiyor')}",
        ],
    )

    print_block(
        "FILES",
        [
            f"Karsilastirma: {comparison_path}",
            f"Tuned metrikler: {tuned_metrics_csv}",
            f"Classification report: {tuned_report_txt}",
            f"Confusion matrix (png): {tuned_cm_png}",
            f"Tuning best params: {tuning_best_json}",
            f"SHAP gorseli: {shap_outputs.get('primary_visual_path', 'yok')}",
            f"Model dosyasi: {tuned_model_path}",
            f"Calisma ozeti: {run_summary_json}",
        ],
    )

    print_block(
        "RESULT",
        [
            "Hafta 6 scripti basariyla tamamlandi.",
            "Detayli raporlar terminal yerine dosyalara kaydedildi.",
            "Sonraki adim: scripts/week7_error_analysis.py",
        ],
    )


if __name__ == "__main__":
    main()
