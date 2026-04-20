from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs" / "week13"
DOCS_DIR = ROOT_DIR / "docs" / "week13"
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
    {
        "id": "edge_case_strong_score_weak_employment",
        "title": "Güçlü Skor, Zayıf İstihdam",
        "expected_decision": "Manuel İnceleme",
        "payload": {
            "Age": 29,
            "Income": 68000,
            "LoanAmount": 30000,
            "CreditScore": 780,
            "MonthsEmployed": 3,
            "NumCreditLines": 3,
            "InterestRate": 13.8,
            "LoanTerm": 48,
            "Education": "Bachelor's",
            "EmploymentType": "Full-time",
            "MaritalStatus": "Single",
            "HasMortgage": "No",
            "HasDependents": "No",
            "LoanPurpose": "Business",
            "HasCoSigner": "No",
        },
    },
]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def decision_label(risk_status: int) -> str:
    return "Onaylanabilir Profil" if risk_status == 0 else "Manuel İnceleme"


def format_percent(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"%{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def http_request(path: str, method: str = "GET", payload: dict[str, Any] | None = None, timeout: float = 10.0) -> dict[str, Any]:
    url = f"{SERVER_BASE_URL}{path}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        status = int(exc.code)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    parsed_json = None
    if body.strip().startswith("{") or body.strip().startswith("["):
        try:
            parsed_json = json.loads(body)
        except json.JSONDecodeError:
            parsed_json = None

    return {
        "path": path,
        "method": method,
        "status": status,
        "length": len(body),
        "elapsed_ms": elapsed_ms,
        "json": parsed_json,
    }


def endpoint_available(path: str = "/", timeout: float = 3.0) -> bool:
    try:
        response = http_request(path, timeout=timeout)
        return 200 <= int(response["status"]) < 300
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

    for _ in range(40):
        if endpoint_available("/"):
            return process
        if process.poll() is not None:
            break
        time.sleep(0.5)

    process.terminate()
    raise RuntimeError("FastAPI server could not be started for Week 13 stress test.")


def run_route_checks() -> list[dict[str, Any]]:
    routes = [
        "/",
        "/genel-bakis",
        "/demo-senaryolari",
        "/model-izleme",
        "/kurallar",
    ]
    rows = []
    for route in routes:
        result = http_request(route)
        rows.append({**result, "passed": 200 <= result["status"] < 300})
    return rows


def run_static_checks() -> list[dict[str, Any]]:
    assets = [
        "/static/style.css?v=20260420-week13",
        "/static/script.js?v=20260420-week13",
        "/static/figures/model_comparison_grouped_bar_week10.png?v=20260420-week13",
        "/static/figures/confusion_matrix_week10.png?v=20260420-week13",
        "/static/figures/shap_false_positive_week10.png?v=20260420-week13",
        "/static/figures/shap_false_negative_week10.png?v=20260420-week13",
    ]
    rows = []
    for asset in assets:
        result = http_request(asset)
        rows.append({**result, "passed": 200 <= result["status"] < 300 and result["length"] > 0})
    return rows


def run_predict_scenarios() -> list[dict[str, Any]]:
    rows = []
    for scenario in SCENARIOS:
        result = http_request("/predict", method="POST", payload=scenario["payload"])
        response_json = result.get("json") or {}
        actual_decision = decision_label(int(response_json.get("risk_durumu", -1))) if response_json else "Yanıt Yok"
        rows.append(
            {
                **result,
                "id": scenario["id"],
                "title": scenario["title"],
                "expected_decision": scenario["expected_decision"],
                "actual_decision": actual_decision,
                "risk_score": response_json.get("temerrut_olasiligi"),
                "model_score": response_json.get("model_temerrut_olasiligi"),
                "passed": result["status"] == 200 and actual_decision == scenario["expected_decision"],
            }
        )
    return rows


def run_invalid_payload_check() -> dict[str, Any]:
    result = http_request("/predict", method="POST", payload={"Age": 17, "Income": 0})
    return {**result, "passed": result["status"] == 422}


def run_repeated_predict_check(iterations: int = 100) -> dict[str, Any]:
    durations: list[float] = []
    errors: list[dict[str, Any]] = []
    payload = SCENARIOS[0]["payload"]

    for idx in range(iterations):
        result = http_request("/predict", method="POST", payload=payload, timeout=15.0)
        durations.append(float(result["elapsed_ms"]))
        if result["status"] != 200:
            errors.append({"iteration": idx + 1, "status": result["status"], "length": result["length"]})

    return {
        "iterations": iterations,
        "passed": not errors,
        "errors": errors,
        "avg_ms": round(statistics.mean(durations), 2),
        "min_ms": round(min(durations), 2),
        "max_ms": round(max(durations), 2),
        "p95_ms": round(sorted(durations)[int(iterations * 0.95) - 1], 2),
    }


def write_report(payload: dict[str, Any]) -> None:
    route_rows = "\n".join(
        f"| {row['method']} {row['path']} | {row['status']} | {row['elapsed_ms']:.2f} | {'Geçti' if row['passed'] else 'Kaldı'} |"
        for row in payload["route_checks"]
    )
    static_rows = "\n".join(
        f"| {row['method']} {row['path']} | {row['status']} | {row['length']} | {'Geçti' if row['passed'] else 'Kaldı'} |"
        for row in payload["static_checks"]
    )
    scenario_rows = "\n".join(
        (
            f"| {row['title']} | {row['expected_decision']} | {row['actual_decision']} | "
            f"{format_percent(row['model_score'])} | {format_percent(row['risk_score'])} | "
            f"{'Geçti' if row['passed'] else 'Kaldı'} |"
        )
        for row in payload["predict_scenarios"]
    )
    repeated = payload["repeated_predict"]
    invalid = payload["invalid_payload"]

    report = f"""# Hafta 13 - Stress Test Raporu

## Genel Sonuç

| Kontrol | Sonuç |
| --- | --- |
| Genel durum | {'Başarılı' if payload['passed'] else 'Kontrol gerekli'} |
| Çalıştırma zamanı | {payload['checked_at']} |
| Test edilen UI route sayısı | {len(payload['route_checks'])} |
| Test edilen static asset sayısı | {len(payload['static_checks'])} |
| Test edilen predict senaryosu | {len(payload['predict_scenarios'])} |
| Ardışık predict isteği | {repeated['iterations']} |

## Route Kontrolleri

| Endpoint | Status | Süre (ms) | Durum |
| --- | ---: | ---: | --- |
{route_rows}

## Static Asset Kontrolleri

| Asset | Status | Boyut | Durum |
| --- | ---: | ---: | --- |
{static_rows}

## Predict Senaryoları

| Senaryo | Beklenen | Gerçek | Model Skoru | Düzeltilmiş Skor | Durum |
| --- | --- | --- | ---: | ---: | --- |
{scenario_rows}

## Invalid Payload Kontrolü

| Endpoint | Beklenen | Gerçek | Durum |
| --- | ---: | ---: | --- |
| POST /predict | 422 | {invalid['status']} | {'Geçti' if invalid['passed'] else 'Kaldı'} |

## Ardışık Predict Stres Kontrolü

| Metrik | Değer |
| --- | ---: |
| İstek sayısı | {repeated['iterations']} |
| Hata sayısı | {len(repeated['errors'])} |
| Ortalama süre | {repeated['avg_ms']} ms |
| Minimum süre | {repeated['min_ms']} ms |
| Maksimum süre | {repeated['max_ms']} ms |
| P95 süre | {repeated['p95_ms']} ms |

## Yorum

Hafta 13 stress testi, CreditScope'un final demo öncesinde route, static asset, API davranışı, edge-case karar mantığı ve ardışık tahmin dayanıklılığı açısından kontrol edildiğini gösterir. Bu test gerçek production load test değildir; akademik demo güvenilirliği için dengeli bir sağlamlaştırma kontrolüdür.
"""
    (DOCS_DIR / "stress_test_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    server_process = ensure_api_server()

    try:
        route_checks = run_route_checks()
        static_checks = run_static_checks()
        predict_scenarios = run_predict_scenarios()
        invalid_payload = run_invalid_payload_check()
        repeated_predict = run_repeated_predict_check()

        payload = {
            "checked_at": datetime.now().isoformat(timespec="seconds"),
            "server_base_url": SERVER_BASE_URL,
            "route_checks": route_checks,
            "static_checks": static_checks,
            "predict_scenarios": predict_scenarios,
            "invalid_payload": invalid_payload,
            "repeated_predict": repeated_predict,
        }
        payload["passed"] = (
            all(row["passed"] for row in route_checks)
            and all(row["passed"] for row in static_checks)
            and all(row["passed"] for row in predict_scenarios)
            and invalid_payload["passed"]
            and repeated_predict["passed"]
        )

        write_json(OUTPUT_DIR / "stress_test_results.json", payload)
        write_report(payload)
    finally:
        if server_process is not None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()

    print("Week 13 stress test complete.")
    print(f"Results: {OUTPUT_DIR / 'stress_test_results.json'}")
    print(f"Report : {DOCS_DIR / 'stress_test_report.md'}")


if __name__ == "__main__":
    main()
