from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

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
        "CREDITSCOPE | HAFTA 7 | HATA ANALIZI, DETAYLI SHAP & RAPOR TASLAGI\n"
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
    raise FileNotFoundError(f"Veri seti bulunamadi. Kontrol edilen yollar:\n{checked}")


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


def generate_error_analysis(x_test: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, week7_dir: Path) -> dict:
    df_analysis = x_test.copy()
    df_analysis["TrueLabel"] = y_true.values
    df_analysis["PredictedLabel"] = y_pred

    # Error Types
    # True Positives: True=1, Pred=1
    # True Negatives: True=0, Pred=0
    # False Positives: True=0, Pred=1
    # False Negatives: True=1, Pred=0

    df_analysis["ErrorType"] = "Unknown"
    df_analysis.loc[(df_analysis["TrueLabel"] == 1) & (df_analysis["PredictedLabel"] == 1), "ErrorType"] = "TP"
    df_analysis.loc[(df_analysis["TrueLabel"] == 0) & (df_analysis["PredictedLabel"] == 0), "ErrorType"] = "TN"
    df_analysis.loc[(df_analysis["TrueLabel"] == 0) & (df_analysis["PredictedLabel"] == 1), "ErrorType"] = "FP"
    df_analysis.loc[(df_analysis["TrueLabel"] == 1) & (df_analysis["PredictedLabel"] == 0), "ErrorType"] = "FN"

    # Save summary 
    summary_stats = df_analysis.groupby("ErrorType").mean().round(2)
    summary_csv_path = week7_dir / "error_analysis_feature_means.csv"
    summary_stats.to_csv(summary_csv_path)

    # Boxplots for numerical vs ErrorType
    numeric_cols = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "InterestRate", "DTIRatio"]
    available_cols = [c for c in numeric_cols if c in df_analysis.columns]
    
    if available_cols:
        fig, axes = plt.subplots(len(available_cols), 1, figsize=(8, 4 * len(available_cols)))
        if len(available_cols) == 1:
            axes = [axes]
        
        for ax, col in zip(axes, available_cols):
            data_to_plot = [
                df_analysis[df_analysis["ErrorType"] == "TP"][col].dropna(),
                df_analysis[df_analysis["ErrorType"] == "FP"][col].dropna(),
                df_analysis[df_analysis["ErrorType"] == "FN"][col].dropna(),
                df_analysis[df_analysis["ErrorType"] == "TN"][col].dropna(),
            ]
            ax.boxplot(data_to_plot, labels=["TP (Dogru Riskli)", "FP (Yanlis Riskli)", "FN (Kacirilan Risk)", "TN (Dogru Guvenilir)"])
            ax.set_title(f"{col} Dagilimi - Hata Tiplerine Gore")
            ax.set_ylabel(col)
        
        plt.tight_layout()
        boxplot_path = week7_dir / "error_analysis_boxplots.png"
        plt.savefig(boxplot_path, dpi=150)
        plt.close()

    error_counts = df_analysis["ErrorType"].value_counts().to_dict()
    
    return {
        "error_counts": error_counts,
        "summary_csv_path": str(summary_csv_path),
        "fp_indices": df_analysis[df_analysis["ErrorType"] == "FP"].index,
        "fn_indices": df_analysis[df_analysis["ErrorType"] == "FN"].index,
        "tp_indices": df_analysis[df_analysis["ErrorType"] == "TP"].index,
        "tn_indices": df_analysis[df_analysis["ErrorType"] == "TN"].index,
    }

def generate_targeted_shap(model, x_test: pd.DataFrame, error_indices: pd.Index, week7_dir: Path, prefix: str):
    if len(error_indices) == 0:
        return None
        
    sample_df = x_test.loc[error_indices]
    # Keep it manageable for SHAP
    if len(sample_df) > 500:
        sample_df = sample_df.sample(n=500, random_state=42)
        
    plot_path = week7_dir / f"shap_summary_{prefix}.png"
    
    try:
        import shap  # type: ignore
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_df)
        
        plt.figure()
        shap.summary_plot(shap_values, sample_df, show=False)
        plt.title(f"SHAP Ozeti - {prefix}")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        return str(plot_path)
    except Exception as exc:
        print(f"SHAP kütüphanesi hatası ({prefix}), XGBoost fallback kullanılıyor: {exc}")
        
        # Fallback to xgboost pred_contribs
        from xgboost import DMatrix
        dmatrix = DMatrix(sample_df, feature_names=sample_df.columns.tolist())
        contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
        contrib_values = contribs[:, :-1]
        mean_abs = np.abs(contrib_values).mean(axis=0)

        importance = pd.Series(mean_abs, index=sample_df.columns).sort_values(ascending=False).head(15)
        
        plt.figure(figsize=(7, 5))
        importance.sort_values().plot(kind="barh", color="#17916c")
        plt.title(f"En Etkili 15 Faktör - {prefix}")
        plt.xlabel("Ortalama Mutlak Etki")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        return str(plot_path)

def generate_report_draft(week7_dir: Path, error_analysis: dict, shap_fn_path: str, shap_fp_path: str):
    report_path = week7_dir / "week7_report_draft.md"
    
    tp = error_analysis["error_counts"].get("TP", 0)
    tn = error_analysis["error_counts"].get("TN", 0)
    fp = error_analysis["error_counts"].get("FP", 0)
    fn = error_analysis["error_counts"].get("FN", 0)
    total = tp + tn + fp + fn
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    content = f"""# CreditScope: Proje Raporu Taslağı (Hafta 7)

## 1. Giriş
Bu analizde, CreditScope XGBoost kredi risk değerlendirme modelimizin detaylı çıktıları incelenmiştir. Ekibimiz (Baran, Arda) özellikle modelin başarısız olduğu ve hata yaptığı durumları analiz ederek SHAP (SHapley Additive exPlanations) tabanlı öngörüler çıkarmıştır.

## 2. Model Hata Matrisi ve Metrikleri
Test seti üzerinde gerçekleştirilen değerlendirmede, modelimizin vakaları aşağıdaki şekilde ayırdığı görülmektedir:

- **True Positive (TP) - Doğru Bildirilen Riskler:** {tp}
- **True Negative (TN) - Doğru Bildirilen Güvenilirler:** {tn}
- **False Positive (FP) - Yanlış Alarm (Riskli Denilen Ama Değil):** {fp}
- **False Negative (FN) - Kaçırılan Riskler (Güvenilir Denilen Ama Riskli):** {fn}

Modelin bankacılık bağlamındaki temel hedefi riskli müşteriyi kaçırmamaktır (Recall Maksimizasyonu). Mevcut Recall değeri **%{recall*100:.2f}**'dir. Ayrıca kesinlik (Precision) ise **%{precision*100:.2f}** seviyesindedir. Yanlış alarm (FP) sayısının yüksek olması, sınıf dengesizliği optimizasyonlarında beklenen bir durumdur.

## 3. Hata Analizi Zıtlaşmaları
- **False Negative (FN) Profili:** Modelin güvenilir bulup gerçekte temerrüde düşen vakalar. Ortalama özellik dağılımları `error_analysis_feature_means.csv` içerisinde incelenmiş ve gözlemlenmiştir ki bu profildeki müşterilerin özellikleri... *(Arda: Buraya CSV'ye bakarak yorum ekle)*.
- **False Positive (FP) Profili:** Modelin riskli bulup gerçekte ödeyen vakalar. Bu durum, modelin katı davrandığı kesimi temsil eder. *(Baran: Buraya Boxplot görsellerine göre yorum ekle).*

## 4. SHAP Odaklı Değerlendirme

Modelin kararlarını neyin yönlendirdiğini anlamak için, yanlılık yaptığı FP ve FN gruplarına özel SHAP grafikleri oluşturulmuştur.

### Kaçırılan Vakalar (False Negatives - SHAP)
Modelin FN tahminlerinde, modeli kararı "0" (Güvenilir) vermeye iten en önemli faktörler şunlardır:
*(Şu görsele bakarak yorumlanacak: `shap_summary_False_Negatives.png`)*

### Yanlış Alarmlar (False Positives - SHAP)
Modelin FP tahminlerinde, karar vericiyi "1" (Riskli) yapmaya iten hatalı ağırlıklandırmaların faktörleri:
*(Şu görsele bakarak yorumlanacak: `shap_summary_False_Positives.png`)*

## 5. Sonuç ve Öneriler
Bu hata analizi ışığında modelin zayıf noktaları tespit edilmiştir. Sonraki aşamalarda (Hafta 9-12 arası) bu zafiyetlerin arayüz üzerinden iş kuralları (business rules) girilerek kompanse edilmesi değerlendirilecektir.

---
*Otomatik Oluşturulma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    report_path.write_text(content, encoding="utf-8")
    return str(report_path)

def main() -> None:
    week7_dir = ROOT_DIR / "outputs" / "week7"
    model_dir = ROOT_DIR / "models"
    week7_dir.mkdir(parents=True, exist_ok=True)
    
    print_header()

    dataset_path = locate_dataset()
    df = pd.read_csv(dataset_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Hedef kolon '{TARGET_COL}' bulunamadi: {dataset_path}")

    # Same Preprocessing as Week 6
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

    model_path = model_dir / "week6_xgboost_tuned.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Week 6 modeli bulunamadi: {model_path}. Lutfen once week6 scriptini calistirin.")

    print_block("INFO", ["Week 6 modeli yukleniyor...", str(model_path)])
    tuned_model = joblib.load(model_path)
    
    # 1. Tahmin
    y_pred = tuned_model.predict(x_test)
    
    # 2. Hata Analizi
    print_block("INFO", ["Hata Vakaları (TP, TN, FP, FN) Ayrıştırılıyor..."])
    error_results = generate_error_analysis(x_test, y_test, y_pred, week7_dir)
    counts = error_results["error_counts"]
    print(f"    TP(Dogru Riskli): {counts.get('TP', 0)}")
    print(f"    TN(Dogru Guvenilir): {counts.get('TN', 0)}")
    print(f"    FP(Yanlis Alarm): {counts.get('FP', 0)}")
    print(f"    FN(Kacirilan): {counts.get('FN', 0)}")
    
    # 3. Odaklı SHAP
    print_block("INFO", ["False Positives ve False Negatives icin Ozel SHAP Grafikleri Üretiliyor..."])
    shap_fn_path = generate_targeted_shap(tuned_model, x_test, error_results["fn_indices"], week7_dir, "False_Negatives")
    shap_fp_path = generate_targeted_shap(tuned_model, x_test, error_results["fp_indices"], week7_dir, "False_Positives")
    
    # 4. Rapor Taslagi
    print_block("INFO", ["Market (Markdown) Formatinda Rapor Taslagi (Draft) Olusturuluyor..."])
    report_path = generate_report_draft(week7_dir, error_results, shap_fn_path, shap_fp_path)
    
    print_block(
        "FILES",
        [
            f"Özellik Ortalama Tablosu: {error_results['summary_csv_path']}",
            f"Kutu Grafikleri (Boxplots): {week7_dir / 'error_analysis_boxplots.png'}",
            f"SHAP False Negatives: {shap_fn_path}",
            f"SHAP False Positives: {shap_fp_path}",
            f"RAPOR TASLAĞI: {report_path}",
        ],
    )

    print_block(
        "RESULT",
        [
            "Hafta 7 scripti basariyla tamamlandi.",
            "Detaylar icin outputs/week7 klasorundeki week7_report_draft.md dosyasini inceleyiniz."
        ],
    )


if __name__ == "__main__":
    main()
