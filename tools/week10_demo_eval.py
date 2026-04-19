from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs" / "week10"
DOCS_DIR = ROOT_DIR / "docs" / "week10"
FIGURES_DIR = DOCS_DIR / "figures"

TRAINING_METRICS_PATH = ROOT_DIR / "outputs" / "training" / "training_metrics.json"
CONFUSION_MATRIX_PATH = ROOT_DIR / "outputs" / "training" / "confusion_matrix.csv"
SHAP_DIR = ROOT_DIR / "outputs" / "shap"

SERVER_BASE_URL = "http://127.0.0.1:8000"

SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "low_risk",
        "title": "Düşük Risk",
        "expected_decision": "Onaylanabilir Profil",
        "payload": {
            "Age": 38,
            "Income": 92000,
            "LoanAmount": 18000,
            "CreditScore": 790,
            "MonthsEmployed": 84,
            "NumCreditLines": 2,
            "InterestRate": 9.8,
            "LoanTerm": 36,
            "Education": "Master's",
            "EmploymentType": "Full-time",
            "MaritalStatus": "Married",
            "HasMortgage": "No",
            "HasDependents": "No",
            "LoanPurpose": "Auto",
            "HasCoSigner": "Yes",
        },
    },
    {
        "id": "manual_review",
        "title": "Manuel İnceleme",
        "expected_decision": "Manuel İnceleme",
        "payload": {
            "Age": 31,
            "Income": 54000,
            "LoanAmount": 29500,
            "CreditScore": 665,
            "MonthsEmployed": 22,
            "NumCreditLines": 4,
            "InterestRate": 16.4,
            "LoanTerm": 48,
            "Education": "Bachelor's",
            "EmploymentType": "Self-employed",
            "MaritalStatus": "Single",
            "HasMortgage": "No",
            "HasDependents": "Yes",
            "LoanPurpose": "Business",
            "HasCoSigner": "No",
        },
    },
    {
        "id": "high_risk",
        "title": "Yüksek Risk",
        "expected_decision": "Manuel İnceleme",
        "payload": {
            "Age": 24,
            "Income": 36000,
            "LoanAmount": 34000,
            "CreditScore": 560,
            "MonthsEmployed": 4,
            "NumCreditLines": 6,
            "InterestRate": 23.5,
            "LoanTerm": 60,
            "Education": "High School",
            "EmploymentType": "Unemployed",
            "MaritalStatus": "Single",
            "HasMortgage": "Yes",
            "HasDependents": "Yes",
            "LoanPurpose": "Other",
            "HasCoSigner": "No",
        },
    },
]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def decision_label(risk_status: int) -> str:
    return "Onaylanabilir Profil" if risk_status == 0 else "Manuel İnceleme"


def call_predict_direct(payload: dict[str, Any]) -> dict[str, Any]:
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    from api import LoanApplication, predict_risk

    return predict_risk(LoanApplication(**payload))


def http_json(url: str, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read().decode("utf-8")
        return {
            "status": response.status,
            "length": len(body),
            "json": json.loads(body) if body.strip().startswith("{") else None,
        }


def endpoint_available(path: str = "/", timeout: float = 3.0) -> bool:
    try:
        request = urllib.request.Request(f"{SERVER_BASE_URL}{path}", method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= int(response.status) < 300
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def ensure_api_server() -> subprocess.Popen | None:
    if endpoint_available("/"):
        return None

    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=ROOT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(30):
        if endpoint_available("/"):
            return process
        if process.poll() is not None:
            break
        time.sleep(0.5)

    process.terminate()
    raise RuntimeError("FastAPI server could not be started for Week 10 UI/API checks.")


def run_demo_predictions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        response = call_predict_direct(scenario["payload"])
        actual_decision = decision_label(int(response["risk_durumu"]))
        rows.append(
            {
                "id": scenario["id"],
                "title": scenario["title"],
                "expected_decision": scenario["expected_decision"],
                "actual_decision": actual_decision,
                "passed": actual_decision == scenario["expected_decision"],
                "input_summary": {
                    "Income": scenario["payload"]["Income"],
                    "LoanAmount": scenario["payload"]["LoanAmount"],
                    "CreditScore": scenario["payload"]["CreditScore"],
                    "MonthsEmployed": scenario["payload"]["MonthsEmployed"],
                    "EmploymentType": scenario["payload"]["EmploymentType"],
                    "HasCoSigner": scenario["payload"]["HasCoSigner"],
                },
                "response": response,
                "comment": scenario_comment(scenario["id"], response),
            }
        )

    return rows


def scenario_comment(scenario_id: str, response: dict[str, Any]) -> str:
    adjusted = response["temerrut_olasiligi"]
    model = response["model_temerrut_olasiligi"]
    dti = response["hesaplanan_dti"]
    rule_count = len(response.get("business_rule_adjustments", []))

    if scenario_id == "low_risk":
        return (
            f"Model skoru %{model:.2f}, iş kuralları sonrası %{adjusted:.2f}. "
            f"DTI {dti:.4f} ve {rule_count} kural etkisi düşük risk kararını destekliyor."
        )
    if scenario_id == "manual_review":
        return (
            f"Model skoru %{model:.2f}, iş kuralları sonrası %{adjusted:.2f}. "
            "Profil orta bantta kaldığı için karar manuel inceleme ekranında tartışılabilir."
        )
    return (
        f"Model skoru %{model:.2f}, iş kuralları sonrası %{adjusted:.2f}. "
        f"DTI {dti:.4f}, düşük kredi notu ve istihdam sinyali yüksek risk davranışını gösteriyor."
    )


def run_api_smoke_test() -> dict[str, Any]:
    smoke: dict[str, Any] = {
        "server_base_url": SERVER_BASE_URL,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "checks": [],
    }

    checks = [
        ("index", "GET", "/", None),
        ("css", "GET", "/static/style.css?v=20260419-week10", None),
        ("js", "GET", "/static/script.js?v=20260419-week10", None),
        ("predict_low_risk", "POST", "/predict", SCENARIOS[0]["payload"]),
    ]

    for name, method, path, payload in checks:
        try:
            result = http_json(f"{SERVER_BASE_URL}{path}", method=method, payload=payload)
            smoke["checks"].append(
                {
                    "name": name,
                    "method": method,
                    "path": path,
                    "status": result["status"],
                    "length": result["length"],
                    "passed": 200 <= int(result["status"]) < 300,
                }
            )
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            smoke["checks"].append(
                {
                    "name": name,
                    "method": method,
                    "path": path,
                    "status": None,
                    "length": 0,
                    "passed": False,
                    "error": str(exc),
                }
            )

    smoke["passed"] = all(check["passed"] for check in smoke["checks"])
    return smoke


def create_polished_confusion_matrix() -> str:
    if not CONFUSION_MATRIX_PATH.exists():
        raise FileNotFoundError(f"Confusion matrix not found: {CONFUSION_MATRIX_PATH}")

    cm_df = pd.read_csv(CONFUSION_MATRIX_PATH, index_col=0)
    output_path = OUTPUT_DIR / "confusion_matrix_week10.png"
    docs_path = FIGURES_DIR / "confusion_matrix_week10.png"

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(cm_df.values, cmap="Blues")
    ax.set_title("CreditScope Karışıklık Matrisi", fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel("Tahmin Edilen Sınıf", fontsize=12)
    ax.set_ylabel("Gerçek Sınıf", fontsize=12)
    ax.set_xticks([0, 1], labels=["Güvenilir (0)", "Riskli (1)"])
    ax.set_yticks([0, 1], labels=["Güvenilir (0)", "Riskli (1)"])

    threshold = cm_df.values.max() / 2
    for row_idx in range(cm_df.shape[0]):
        for col_idx in range(cm_df.shape[1]):
            value = int(cm_df.iloc[row_idx, col_idx])
            color = "white" if value > threshold else "#172033"
            ax.text(col_idx, row_idx, f"{value:,}", ha="center", va="center", color=color, fontsize=14, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Başvuru sayısı", rotation=270, labelpad=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    fig.savefig(docs_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return str(docs_path.relative_to(ROOT_DIR))


def create_model_comparison_chart(metrics_payload: dict[str, Any]) -> str:
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split

    from preprocessing import prepare_train_test

    x_train, x_test, y_train, y_test, _, _ = prepare_train_test()
    max_train_rows = 80000
    if len(x_train) > max_train_rows:
        x_train_eval, _, y_train_eval, _ = train_test_split(
            x_train,
            y_train,
            train_size=max_train_rows,
            random_state=42,
            stratify=y_train,
        )
    else:
        x_train_eval, y_train_eval = x_train, y_train

    comparison_models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            max_depth=12,
            n_estimators=160,
            n_jobs=-1,
            random_state=42,
        ),
    }

    rows: list[dict[str, Any]] = []
    for model_name, model in comparison_models.items():
        model.fit(x_train_eval, y_train_eval)
        y_pred = model.predict(x_test)
        rows.append(
            {
                "model": model_name,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "training_note": f"Stratified {len(x_train_eval)}-row training sample",
            }
        )

    xgb_metrics = metrics_payload["metrics"]
    rows.append(
        {
            "model": "XGBoost",
            "accuracy": float(xgb_metrics["accuracy"]),
            "precision": float(xgb_metrics["precision"]),
            "recall": float(xgb_metrics["recall"]),
            "f1": float(xgb_metrics["f1"]),
            "training_note": "SMOTE + tuned params + calibrated threshold",
        }
    )

    metrics_df = pd.DataFrame(rows)
    metrics_csv = OUTPUT_DIR / "model_comparison_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    plot_df = metrics_df.set_index("model")[["accuracy", "precision", "recall", "f1"]]
    output_path = OUTPUT_DIR / "model_comparison_grouped_bar_week10.png"
    docs_path = FIGURES_DIR / "model_comparison_grouped_bar_week10.png"

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    colors = ["#457b9d", "#f4a261", "#2a9d8f", "#7c3aed"]
    plot_df.plot(kind="bar", ax=ax, color=colors, width=0.74)
    ax.set_title("Model Performans Karşılaştırması", fontsize=16, fontweight="bold", pad=16)
    ax.set_ylabel("Skor")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.legend(["Accuracy", "Precision", "Recall", "F1-score"], loc="upper right", frameon=True)
    ax.tick_params(axis="x", rotation=0)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    fig.savefig(docs_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return str(docs_path.relative_to(ROOT_DIR))


def copy_shap_figures() -> dict[str, str]:
    sources = {
        "false_positive": SHAP_DIR / "shap_summary_False_Positives.png",
        "false_negative": SHAP_DIR / "shap_summary_False_Negatives.png",
    }
    targets: dict[str, str] = {}

    for key, source in sources.items():
        if not source.exists():
            raise FileNotFoundError(f"SHAP figure not found: {source}")
        target = FIGURES_DIR / f"shap_{key}_week10.png"
        shutil.copy2(source, target)
        targets[key] = str(target.relative_to(ROOT_DIR))

    return targets


def capture_ui_screenshot() -> str | None:
    if not endpoint_available("/"):
        raise RuntimeError("UI screenshot skipped because FastAPI server is not reachable.")

    candidates = [
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    ]
    browser = next((path for path in candidates if path.exists()), None)
    if browser is None:
        return None

    output_path = OUTPUT_DIR / "ui_cockpit_week10.png"
    docs_path = FIGURES_DIR / "ui_cockpit_week10.png"
    command = [
        str(browser),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--window-size=1440,1100",
        f"--screenshot={output_path}",
        f"{SERVER_BASE_URL}/?v=week10",
    ]
    subprocess.run(command, cwd=ROOT_DIR, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if output_path.stat().st_size < 50000:
        raise RuntimeError(
            f"UI screenshot looks too small ({output_path.stat().st_size} bytes). "
            "The captured page may be a browser error page."
        )
    shutil.copy2(output_path, docs_path)
    return str(docs_path.relative_to(ROOT_DIR))


def write_reports(
    metrics_payload: dict[str, Any],
    demo_predictions: list[dict[str, Any]],
    smoke_test: dict[str, Any],
    figure_paths: dict[str, str],
) -> None:
    metrics = metrics_payload["metrics"]
    threshold = metrics_payload["decision_threshold"]
    counts = load_json(ROOT_DIR / "outputs" / "shap" / "shap_error_analysis_summary.json")["group_analysis"]["counts"]
    comparison_df = pd.read_csv(OUTPUT_DIR / "model_comparison_metrics.csv")
    comparison_rows = "\n".join(
        (
            f"| {row.model} | {row.accuracy:.4f} | {row.precision:.4f} | "
            f"{row.recall:.4f} | {row.f1:.4f} |"
        )
        for row in comparison_df.itertuples(index=False)
    )

    results_md = f"""# Hafta 10 - Results ve Discussion

## Model Sonuçları

| Metrik | Değer |
| --- | ---: |
| Accuracy | {metrics['accuracy']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1-score | {metrics['f1']:.4f} |
| Decision Threshold | {threshold:.3f} |

CreditScope modeli recall odaklı kalibre edilmiştir. Bankacılık senaryosunda temerrüde düşebilecek bir müşteriyi kaçırmak, güvenilir bir müşteriyi manuel incelemeye göndermekten daha maliyetli kabul edildiği için karar eşiği klasik `0.50` yerine `{threshold:.3f}` olarak belirlenmiştir.

## Model Karşılaştırması

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
{comparison_rows}

Logistic Regression ve Random Forest karşılaştırmaları aynı test seti üzerinde hızlı değerlendirme amacıyla stratified eğitim örneklemiyle çalıştırılmıştır. XGBoost satırı ise SMOTE, optimize hiperparametreler ve kalibre karar eşiği ile elde edilen ana CreditScope modelidir. Grafik, modeller arasındaki accuracy/precision/recall/F1 dengesini gösterir; CreditScope demosunda XGBoost seçimi yalnızca tek metrik üstünlüğüne değil, tuning süreci, SHAP uyumu ve API'ye alınmış üretim artifact'lerine dayanır.

## Hata Analizi

| Grup | Adet |
| --- | ---: |
| True Negative | {counts.get('True_Negative', 0)} |
| False Positive | {counts.get('False_Positive', 0)} |
| True Positive | {counts.get('True_Positive', 0)} |
| False Negative | {counts.get('False_Negative', 0)} |

False Positive vakaları modelin temkinli davrandığı başvuruları gösterir. Bu vakalar kredi uzmanı tarafından manuel incelemeye alınabilir ve iş kuralları bu gruptaki gereksiz alarmları azaltmak için kullanılır. False Negative vakaları ise modelin kaçırdığı riskli profilleri temsil eder; bu nedenle SHAP analizi özellikle bu gruptaki karar sinyallerini anlamak için önemlidir.

## Revize Grafikler

- UI kokpit ekran görüntüsü: `figures/ui_cockpit_week10.png`
- Model karşılaştırma grafiği: `figures/model_comparison_grouped_bar_week10.png`
- Karışıklık matrisi: `figures/confusion_matrix_week10.png`
- False Positive SHAP: `figures/shap_false_positive_week10.png`
- False Negative SHAP: `figures/shap_false_negative_week10.png`

![Hafta 10 Risk Kokpiti](figures/ui_cockpit_week10.png)

![Model Karşılaştırması](figures/model_comparison_grouped_bar_week10.png)

![Karışıklık Matrisi](figures/confusion_matrix_week10.png)

## Business Rules Değerlendirmesi

API katmanındaki business rules modeli değiştirmeden karar olasılığını şeffaf şekilde düzeltir. Örneğin güçlü kredi notu, düşük DTI, uzun süreli tam zamanlı istihdam ve kefil varlığı riski düşürürken; yüksek DTI, düşük kredi notu ve istikrarsız istihdam riski artırır. `/predict` yanıtında hem ham model skoru hem de düzeltilmiş skor ayrı verildiği için bu katman denetlenebilir kalır.

## UI Demo Değerlendirmesi

Hafta 9 arayüzü Hafta 10 demo testinde risk kokpiti olarak kullanılmıştır. Arayüz, üç senaryo butonu ile formu doldurur, DTI ve ödeme önizlemelerini canlı hesaplar, API sonucunu karar paneline yansıtır ve business rule etkilerini kullanıcıya açıkça gösterir.
"""

    demo_rows = "\n".join(
        (
            f"| {row['title']} | {row['expected_decision']} | {row['actual_decision']} | "
            f"%{row['response']['model_temerrut_olasiligi']:.2f} | "
            f"%{row['response']['temerrut_olasiligi']:.2f} | "
            f"{'Geçti' if row['passed'] else 'Kontrol gerekli'} |"
        )
        for row in demo_predictions
    )
    smoke_rows = "\n".join(
        f"| {check['name']} | {check['method']} {check['path']} | {check.get('status', '-')} | {'Geçti' if check['passed'] else 'Kaldı'} |"
        for check in smoke_test["checks"]
    )

    demo_md = f"""# Hafta 10 - Demo Test Raporu

## Senaryo Sonuçları

| Senaryo | Beklenen Karar | Gerçek Karar | Model Skoru | Düzeltilmiş Skor | Durum |
| --- | --- | --- | ---: | ---: | --- |
{demo_rows}

## Senaryo Yorumları

{scenario_sections(demo_predictions)}

## API ve Statik Dosya Kontrolleri

| Kontrol | Endpoint | Status | Durum |
| --- | --- | ---: | --- |
{smoke_rows}

## UI Kabul Kriterleri

- Üç demo senaryosu formu otomatik doldurur.
- DTI, kredi/gelir oranı, aylık taksit ve kredi segmenti anlık hesaplanır.
- `/predict` sonucu sağ karar paneline yansır.
- Model skoru, düzeltilmiş skor, eşik ve DTI ayrı gösterilir.
- Business rule etkileri listelenir.

## Kanıt Dosyaları

- Demo tahminleri: `../../outputs/week10/demo_predictions.json`
- API smoke test: `../../outputs/week10/api_smoke_test.json`
- Model metrikleri: `../../outputs/week10/model_metrics.json`
- Figürler: `{figure_paths.get('ui_screenshot', 'UI ekran görüntüsü alınamadı')}`, `{figure_paths['model_comparison']}`, `{figure_paths['confusion_matrix']}`, `{figure_paths['false_positive']}`, `{figure_paths['false_negative']}`
"""

    (DOCS_DIR / "results_discussion.md").write_text(results_md, encoding="utf-8")
    (DOCS_DIR / "demo_test_report.md").write_text(demo_md, encoding="utf-8")


def scenario_sections(demo_predictions: list[dict[str, Any]]) -> str:
    sections = []
    for row in demo_predictions:
        summary = row["input_summary"]
        response = row["response"]
        rules = ", ".join(rule["id"] for rule in response.get("business_rule_adjustments", [])) or "Yok"
        sections.append(
            f"""### {row['title']}

- Gelir: `{summary['Income']}`, kredi tutarı: `{summary['LoanAmount']}`, kredi notu: `{summary['CreditScore']}`
- İstihdam: `{summary['EmploymentType']}`, çalışma süresi: `{summary['MonthsEmployed']}` ay, kefil: `{summary['HasCoSigner']}`
- API kararı: **{row['actual_decision']}**
- Uygulanan kurallar: {rules}
- Yorum: {row['comment']}
"""
        )
    return "\n".join(sections)


def main() -> None:
    ensure_dirs()
    server_process = ensure_api_server()

    try:
        metrics_payload = load_json(TRAINING_METRICS_PATH)
        write_json(OUTPUT_DIR / "model_metrics.json", metrics_payload)

        demo_predictions = run_demo_predictions()
        write_json(OUTPUT_DIR / "demo_predictions.json", demo_predictions)

        smoke_test = run_api_smoke_test()
        write_json(OUTPUT_DIR / "api_smoke_test.json", smoke_test)

        figure_paths = copy_shap_figures()
        figure_paths["model_comparison"] = create_model_comparison_chart(metrics_payload)
        figure_paths["confusion_matrix"] = create_polished_confusion_matrix()
        ui_screenshot = capture_ui_screenshot()
        if ui_screenshot:
            figure_paths["ui_screenshot"] = ui_screenshot

        write_reports(metrics_payload, demo_predictions, smoke_test, figure_paths)
    finally:
        if server_process is not None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()

    print("Week 10 demo evaluation complete.")
    print(f"Demo predictions: {OUTPUT_DIR / 'demo_predictions.json'}")
    print(f"API smoke test  : {OUTPUT_DIR / 'api_smoke_test.json'}")
    print(f"Docs            : {DOCS_DIR}")
    print(f"Figures         : {FIGURES_DIR}")


if __name__ == "__main__":
    main()
