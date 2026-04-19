from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET_COL = "Default"
DATASET_CANDIDATES = [
    ROOT_DIR / "processed_data" / "week3_processed_dataset.csv",
    ROOT_DIR / "week3_processed_dataset.csv",
]


def print_section(title: str) -> None:
    line = "=" * 88
    print(f"\n{line}\n{title}\n{line}")


def locate_input_dataset() -> Path:
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path
    searched = "\n".join(f"- {path}" for path in DATASET_CANDIDATES)
    raise FileNotFoundError(f"Hafta 4 girdi veri seti bulunamadı. Kontrol edilen yollar:\n{searched}")


def apply_feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    engineered_cols: list[str] = []

    if {"Income", "LoanAmount"}.issubset(df.columns):
        col = "Income_minus_LoanAmount"
        df[col] = df["Income"] - df["LoanAmount"]
        engineered_cols.append(col)

    if {"CreditScore", "DTIRatio"}.issubset(df.columns):
        col = "CreditScore_x_DTIRatio"
        df[col] = df["CreditScore"] * df["DTIRatio"]
        engineered_cols.append(col)

    if {"MonthsEmployed", "NumCreditLines"}.issubset(df.columns):
        col = "Employment_per_CreditLine"
        df[col] = df["MonthsEmployed"] / (df["NumCreditLines"] + 1.0)
        engineered_cols.append(col)

    return df, engineered_cols


def evaluate_model(
    model,
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    print(f"{model_name} modeli eğitiliyor...")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = build_classification_report_tr(report_dict)

    print(f"\n{model_name} Metrikleri")
    print(f"Doğruluk   : {metrics['accuracy']:.4f}")
    print(f"Kesinlik   : {metrics['precision']:.4f}")
    print(f"Duyarlılık : {metrics['recall']:.4f}")
    print(f"F1 Skoru            : {metrics['f1']:.4f}")
    print(f"\n{model_name} Sınıflandırma Raporu")
    print(report_text)

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
    }


def save_confusion_matrix_plot(cm, model_name: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title(f"{model_name} Karmaşıklık Matrisi")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_classification_report_tr(report_dict: dict) -> str:
    label_map = {
        "0": "Sınıf 0",
        "1": "Sınıf 1",
        "accuracy": "Genel Doğruluk",
        "macro avg": "Makro Ortalama",
        "weighted avg": "Ağırlıklı Ortalama",
    }

    lines = []
    lines.append("Etiket                 Kesinlik    Duyarlılık    F1 Skoru     Destek")
    lines.append("-" * 70)

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
        lines.append("-" * 70)
        lines.append(
            f"{label_map['accuracy']:<22} {'-':>9} {'-':>13} {accuracy:>11.4f} {support:>10d}"
        )

    return "\n".join(lines)


def main() -> None:
    output_dir = ROOT_DIR / "outputs" / "week4"
    output_dir.mkdir(parents=True, exist_ok=True)

    print_section("HAFTA 4 - OZELLIK MUHENDISLIGI + TEMEL MODELLER")

    dataset_path = locate_input_dataset()
    df = pd.read_csv(dataset_path)
    print(f"Girdi veri seti: {dataset_path}")
    print(f"Girdi boyutu: {df.shape}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Hedef kolon '{TARGET_COL}' bulunamadı: {dataset_path}")

    print_section("OZELLIK MUHENDISLIGI")
    df, engineered_cols = apply_feature_engineering(df.copy())
    if engineered_cols:
        print("Özellik mühendisliği tamamlandı")
        print(f"Yeni üretilen özellikler ({len(engineered_cols)}): {engineered_cols}")
    else:
        print("Özellik mühendisliği tamamlandı")
        print("Ek özellik üretilmedi.")

    y = df[TARGET_COL]
    x = df.drop(columns=[TARGET_COL])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print_section("EGITIM / TEST AYRIMI")
    print(f"Eğitim boyutu / Test boyutu: {x_train.shape} / {x_test.shape}")
    print(f"Eğitim hedef dağılımı: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Test hedef dağılımı  : {y_test.value_counts().sort_index().to_dict()}")

    models = {
        "Lojistik Regresyon": LogisticRegression(
            random_state=42,
            max_iter=1200,
        ),
        "Rastgele Orman": RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            n_jobs=-1,
        ),
    }
    file_slug_map = {
        "Lojistik Regresyon": "lojistik_regresyon",
        "Rastgele Orman": "rastgele_orman",
    }

    results: dict[str, dict] = {}
    print_section("MODEL EGITIMI + DEGERLENDIRME")
    for model_name, model in models.items():
        results[model_name] = evaluate_model(model, model_name, x_train, y_train, x_test, y_test)

    metrics_df = pd.DataFrame([results[name]["metrics"] for name in results]).sort_values(
        by=["recall", "f1"],
        ascending=False,
    )

    metrics_csv_path = output_dir / "week4_baseline_metrics.csv"
    metrics_txt_path = output_dir / "week4_baseline_metrics.txt"
    summary_json_path = output_dir / "week4_run_summary.json"

    metrics_df.to_csv(metrics_csv_path, index=False)
    metrics_txt_path.write_text(metrics_df.to_string(index=False), encoding="utf-8")

    for model_name, payload in results.items():
        slug = file_slug_map.get(model_name, model_name.lower().replace(" ", "_"))
        report_txt = output_dir / f"{slug}_classification_report.txt"
        report_csv = output_dir / f"{slug}_classification_report.csv"
        cm_csv = output_dir / f"{slug}_confusion_matrix.csv"
        cm_png = output_dir / f"{slug}_confusion_matrix.png"

        report_txt.write_text(payload["classification_report_text"], encoding="utf-8")
        pd.DataFrame(payload["classification_report_dict"]).transpose().to_csv(report_csv)
        pd.DataFrame(
            payload["confusion_matrix"],
            index=["actual_0", "actual_1"],
            columns=["pred_0", "pred_1"],
        ).to_csv(cm_csv)
        save_confusion_matrix_plot(payload["confusion_matrix"], model_name, cm_png)

    summary_payload = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_dataset": str(dataset_path),
        "input_shape": list(df.shape),
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "target_distribution_train": y_train.value_counts().sort_index().to_dict(),
        "target_distribution_test": y_test.value_counts().sort_index().to_dict(),
        "engineered_features": engineered_cols,
        "metrics_file": str(metrics_csv_path),
        "output_dir": str(output_dir),
    }
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics_display = metrics_df.rename(
        columns={
            "model": "Model",
            "accuracy": "Doğruluk",
            "precision": "Kesinlik",
            "recall": "Duyarlılık",
            "f1": "F1 Skoru",
        }
    )

    print_section("TEMEL MODEL METRIK OZETI")
    print(metrics_display.to_string(index=False))

    print_section("CIKTILAR KAYDEDILDI")
    print(f"Metrik CSV dosyası: {metrics_csv_path}")
    print(f"Metrik TXT dosyası: {metrics_txt_path}")
    print(f"Çalışma özeti: {summary_json_path}")
    print(f"Detaylı raporlar ve karmaşıklık matrisleri: {output_dir}")

    print_section("HAFTA 4 TAMAMLANDI")
    print("Temel model karşılaştırması tamamlandı")


if __name__ == "__main__":
    main()
