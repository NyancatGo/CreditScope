from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import joblib
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
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET_COL = "Default"
DATASET_CANDIDATES = [
    ROOT_DIR / "processed_data" / "week4_feature_engineered_dataset.csv",
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
    raise FileNotFoundError(f"Hafta 5 girdi veri seti bulunamadı. Kontrol edilen yollar:\n{searched}")


def apply_feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    engineered_cols: list[str] = []

    if "Income_minus_LoanAmount" not in df.columns and {"Income", "LoanAmount"}.issubset(df.columns):
        df["Income_minus_LoanAmount"] = df["Income"] - df["LoanAmount"]
        engineered_cols.append("Income_minus_LoanAmount")

    if "CreditScore_x_DTIRatio" not in df.columns and {"CreditScore", "DTIRatio"}.issubset(df.columns):
        df["CreditScore_x_DTIRatio"] = df["CreditScore"] * df["DTIRatio"]
        engineered_cols.append("CreditScore_x_DTIRatio")

    if "Employment_per_CreditLine" not in df.columns and {"MonthsEmployed", "NumCreditLines"}.issubset(df.columns):
        df["Employment_per_CreditLine"] = df["MonthsEmployed"] / (df["NumCreditLines"] + 1.0)
        engineered_cols.append("Employment_per_CreditLine")

    return df, engineered_cols


def class_distribution_text(y: pd.Series) -> str:
    dist = y.value_counts().sort_index()
    total = int(dist.sum())
    lines = []
    for cls, count in dist.items():
        ratio = (count / total) * 100
        lines.append(f"Sınıf {cls}: {int(count)} satır (%{ratio:.2f})")
    return "\n".join(lines)


def build_report_text_tr(report_dict: dict) -> str:
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


def random_oversample_training_set(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    train_df = x_train.copy()
    train_df[TARGET_COL] = y_train.values

    class_counts = train_df[TARGET_COL].value_counts()
    major_label = class_counts.idxmax()
    minor_label = class_counts.idxmin()

    major_df = train_df[train_df[TARGET_COL] == major_label]
    minor_df = train_df[train_df[TARGET_COL] == minor_label]

    minor_upsampled = resample(
        minor_df,
        replace=True,
        n_samples=len(major_df),
        random_state=random_state,
    )

    balanced_df = pd.concat([major_df, minor_upsampled], axis=0)
    balanced_df = balanced_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    y_bal = balanced_df[TARGET_COL]
    x_bal = balanced_df.drop(columns=[TARGET_COL])
    return x_bal, y_bal


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = build_report_text_tr(report_dict)
    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "report_dict": report_dict,
        "report_text": report_text,
    }


def save_confusion_matrix_plot(cm, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_metrics_block(block_title: str, metrics: dict, report_text: str) -> None:
    print(f"\n{block_title}")
    print(f"Doğruluk   : {metrics['accuracy']:.4f}")
    print(f"Kesinlik   : {metrics['precision']:.4f}")
    print(f"Duyarlılık : {metrics['recall']:.4f}")
    print(f"F1 Skoru   : {metrics['f1']:.4f}")
    print("\nSınıflandırma Raporu")
    print(report_text)


def main() -> None:
    output_dir = ROOT_DIR / "outputs" / "week5"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    print_section("HAFTA 5 - XGBOOST + SINIF DENGESIZLIGI COZUMU")

    dataset_path = locate_input_dataset()
    df = pd.read_csv(dataset_path)
    print(f"Girdi veri seti: {dataset_path}")
    print(f"Girdi boyutu: {df.shape}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Hedef kolon '{TARGET_COL}' bulunamadı: {dataset_path}")

    df, engineered_cols = apply_feature_engineering(df.copy())
    if engineered_cols:
        print(f"Ek özellik üretildi: {engineered_cols}")
    else:
        print("Ek özellik üretimi yapılmadı (özellikler zaten mevcut veya uygun kolon yok).")

    y = df[TARGET_COL]
    x = df.drop(columns=[TARGET_COL])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print_section("SINIF DAGILIMI (DENGELEME ONCESI)")
    print("Eğitim verisinde sınıf dağılımı:")
    print(class_distribution_text(y_train))

    print_section("XGBOOST EGITIMI (DENGESIZ VERI - REFERANS)")
    xgb_base = XGBClassifier(
        random_state=42,
        n_estimators=220,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb_base.fit(x_train, y_train)
    base_result = evaluate_model(xgb_base, x_test, y_test)
    print_metrics_block("Referans (dengesiz) sonuçlar", base_result["metrics"], base_result["report_text"])

    print_section("DENGELEME ADIMI")
    try:
        from imblearn.over_sampling import SMOTE  # type: ignore

        print("SMOTE uygulanıyor...")
        sampler_name = "SMOTE"
        sampler = SMOTE(random_state=42)
        x_train_bal, y_train_bal = sampler.fit_resample(x_train, y_train)
    except Exception:
        print("SMOTE kütüphanesi bulunamadı. SMOTE alternatifi: Rastgele Aşırı Örnekleme uygulanıyor...")
        sampler_name = "RandomOverSampling"
        x_train_bal, y_train_bal = random_oversample_training_set(x_train, y_train, random_state=42)

    print("\nSınıf dağılımı (denge sonrası):")
    print(class_distribution_text(pd.Series(y_train_bal)))

    print_section("XGBOOST EGITIMI (DENGELENMIS VERI)")
    xgb_balanced = XGBClassifier(
        random_state=42,
        n_estimators=220,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb_balanced.fit(x_train_bal, y_train_bal)
    balanced_result = evaluate_model(xgb_balanced, x_test, y_test)
    print_metrics_block("Dengelenmiş sonuçlar", balanced_result["metrics"], balanced_result["report_text"])

    print_section("RECALL IYILESME ANALIZI")
    recall_before = base_result["metrics"]["recall"]
    recall_after = balanced_result["metrics"]["recall"]
    recall_delta = recall_after - recall_before

    print(f"Dengeleme öncesi Recall : {recall_before:.4f}")
    print(f"Dengeleme sonrası Recall: {recall_after:.4f}")
    print(f"Recall değişimi         : {recall_delta:+.4f}")

    if recall_delta >= 0:
        print("Recall iyileşmesi elde edildi.")
    else:
        print("Recall düşüşü gözlendi. Parametre ayarı/tuning ile iyileştirme önerilir.")

    rows = [
        {
            "senaryo": "xgboost_dengesiz",
            **base_result["metrics"],
        },
        {
            "senaryo": f"xgboost_{sampler_name.lower()}",
            **balanced_result["metrics"],
        },
    ]
    metrics_df = pd.DataFrame(rows)

    metrics_csv = output_dir / "week5_xgboost_metrics_comparison.csv"
    metrics_txt = output_dir / "week5_xgboost_metrics_comparison.txt"
    recall_txt = output_dir / "week5_recall_analysis.txt"
    summary_json = output_dir / "week5_run_summary.json"

    metrics_df.to_csv(metrics_csv, index=False)
    metrics_txt.write_text(metrics_df.to_string(index=False), encoding="utf-8")
    recall_txt.write_text(
        (
            "Hafta 5 Recall Analizi\n"
            f"- Dengeleme yöntemi: {sampler_name}\n"
            f"- Recall (önce): {recall_before:.6f}\n"
            f"- Recall (sonra): {recall_after:.6f}\n"
            f"- Recall değişimi: {recall_delta:+.6f}\n"
        ),
        encoding="utf-8",
    )

    artifact_map = {
        "xgboost_dengesiz": base_result,
        f"xgboost_{sampler_name.lower()}": balanced_result,
    }
    for name, payload in artifact_map.items():
        report_txt = output_dir / f"{name}_classification_report.txt"
        report_csv = output_dir / f"{name}_classification_report.csv"
        cm_csv = output_dir / f"{name}_confusion_matrix.csv"
        cm_png = output_dir / f"{name}_confusion_matrix.png"

        report_txt.write_text(payload["report_text"], encoding="utf-8")
        pd.DataFrame(payload["report_dict"]).transpose().to_csv(report_csv)
        pd.DataFrame(
            payload["confusion_matrix"],
            index=["gercek_0", "gercek_1"],
            columns=["tahmin_0", "tahmin_1"],
        ).to_csv(cm_csv)
        save_confusion_matrix_plot(payload["confusion_matrix"], f"{name} - Karmaşıklık Matrisi", cm_png)

    model_path = model_dir / f"week5_xgboost_{sampler_name.lower()}.pkl"
    joblib.dump(xgb_balanced, model_path)

    summary_payload = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_dataset": str(dataset_path),
        "input_shape": list(df.shape),
        "engineered_features": engineered_cols,
        "sampler_method": sampler_name,
        "before_balance_distribution": y_train.value_counts().sort_index().to_dict(),
        "after_balance_distribution": pd.Series(y_train_bal).value_counts().sort_index().to_dict(),
        "recall_before": recall_before,
        "recall_after": recall_after,
        "recall_delta": recall_delta,
        "metrics_file": str(metrics_csv),
        "model_file": str(model_path),
    }
    summary_json.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_section("CIKTILAR KAYDEDILDI")
    print(f"Metrik karşılaştırması (CSV): {metrics_csv}")
    print(f"Metrik karşılaştırması (TXT): {metrics_txt}")
    print(f"Recall analizi: {recall_txt}")
    print(f"Çalışma özeti: {summary_json}")
    print(f"Kaydedilen model: {model_path}")
    print(f"Detaylı raporlar ve karmaşıklık matrisleri: {output_dir}")

    print_section("HAFTA 5 TAMAMLANDI")
    print("XGBoost değerlendirmesi ve sınıf dengesizliği çözümü tamamlandı.")


if __name__ == "__main__":
    main()
