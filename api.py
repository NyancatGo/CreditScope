from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from preprocessing import ROOT_DIR, prepare_inference_frame


MODEL_PATH = ROOT_DIR / "xgboost_optimized.pkl"
SCALER_PATH = ROOT_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = ROOT_DIR / "feature_names.pkl"
DECISION_THRESHOLD_PATH = ROOT_DIR / "decision_threshold.pkl"
TEMPLATES_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"
BUSINESS_RULE_VERSION = "default-review-2026-04-19"
ASSET_VERSION = "20260419-week12"


app = FastAPI(title="CreditScope - Risk Analizi API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals["asset_version"] = ASSET_VERSION

print("Loading CreditScope model artifacts...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)
decision_threshold = joblib.load(DECISION_THRESHOLD_PATH) if DECISION_THRESHOLD_PATH.exists() else 0.50


# --- Static snapshots (week 10/11 validated results) ---------------------------------

DEMO_SCENARIOS = [
    {
        "key": "safe",
        "title": "Düşük Risk",
        "expected": "Onaylanabilir Profil",
        "actual": "Onaylanabilir Profil",
        "model_score": 8.24,
        "adjusted_score": 4.81,
        "dti": 0.1957,
        "profile": {
            "Gelir": "92.000 $",
            "Kredi Tutarı": "18.000 $",
            "Kredi Notu": 790,
            "İstihdam": "Full-time",
            "Çalışma Süresi": "84 ay",
            "Kefil": "Var",
        },
        "applied_rules": [
            "strong_credit_score",
            "low_dti",
            "stable_full_time_employment",
            "cosigner_present",
        ],
        "comment": (
            "Model skoru %8.24, iş kuralları sonrası %4.81. DTI 0.1957 ve dört pozitif "
            "kuralın birlikte etkisi düşük risk kararını destekliyor."
        ),
        "status": "passed",
    },
    {
        "key": "review",
        "title": "Manuel İnceleme",
        "expected": "Manuel İnceleme",
        "actual": "Manuel İnceleme",
        "model_score": 40.01,
        "adjusted_score": 48.02,
        "dti": 0.5463,
        "profile": {
            "Gelir": "54.000 $",
            "Kredi Tutarı": "29.500 $",
            "Kredi Notu": 665,
            "İstihdam": "Self-employed",
            "Çalışma Süresi": "22 ay",
            "Kefil": "Yok",
        },
        "applied_rules": ["high_dti"],
        "comment": (
            "Model skoru %40.01, iş kuralları sonrası %48.02. Profil orta bantta olduğu "
            "için karar manuel inceleme olarak şekilleniyor."
        ),
        "status": "passed",
    },
    {
        "key": "risky",
        "title": "Yüksek Risk",
        "expected": "Manuel İnceleme",
        "actual": "Manuel İnceleme",
        "model_score": 58.04,
        "adjusted_score": 92.11,
        "dti": 0.9444,
        "profile": {
            "Gelir": "36.000 $",
            "Kredi Tutarı": "34.000 $",
            "Kredi Notu": 560,
            "İstihdam": "Unemployed",
            "Çalışma Süresi": "4 ay",
            "Kefil": "Yok",
        },
        "applied_rules": ["high_dti", "weak_credit_score", "unstable_employment"],
        "comment": (
            "Model skoru %58.04, iş kuralları sonrası %92.11. Yüksek DTI, düşük kredi "
            "notu ve istihdam sinyali birlikte yüksek risk davranışını gösteriyor."
        ),
        "status": "passed",
    },
    {
        "key": "edge",
        "title": "Güçlü Skor, Zayıf İstihdam",
        "expected": "Manuel İnceleme",
        "actual": "Manuel İnceleme",
        "model_score": 26.68,
        "adjusted_score": 24.54,
        "dti": 0.4412,
        "profile": {
            "Gelir": "68.000 $",
            "Kredi Tutarı": "30.000 $",
            "Kredi Notu": 780,
            "İstihdam": "Full-time",
            "Çalışma Süresi": "3 ay",
            "Kefil": "Yok",
        },
        "applied_rules": ["strong_credit_score", "unstable_employment"],
        "comment": (
            "Model skoru %26.68, iş kuralları sonrası %24.54. Güçlü kredi notuna rağmen "
            "kısa çalışma geçmişi profili manuel inceleme bandında tutuyor."
        ),
        "status": "passed",
    },
]

FINAL_METRICS = {
    "accuracy": 0.6827,
    "precision": 0.2217,
    "recall": 0.6901,
    "f1": 0.3356,
    "threshold": 0.231,
    "recall_target": 0.69,
    "recall_floor": 0.67,
}

MODEL_COMPARISON = [
    {"name": "Logistic Regression", "accuracy": 0.6885, "precision": 0.2265, "recall": 0.6962, "f1": 0.3417},
    {"name": "Random Forest", "accuracy": 0.8117, "precision": 0.2968, "recall": 0.4539, "f1": 0.3589},
    {"name": "XGBoost (Final)", "accuracy": 0.6827, "precision": 0.2217, "recall": 0.6901, "f1": 0.3356},
]

CONFUSION_MATRIX = {
    "true_negative": 30773,
    "false_positive": 14366,
    "true_positive": 4093,
    "false_negative": 1838,
}

BUSINESS_RULES = [
    {
        "id": "strong_credit_score",
        "label": "Güçlü Kredi Notu",
        "condition": "CreditScore ≥ 750",
        "multiplier": 0.80,
        "impact": "Temerrüt olasılığını %20 azalt",
        "direction": "reduce",
    },
    {
        "id": "low_dti",
        "label": "Düşük DTI",
        "condition": "DTIRatio ≤ 0.35",
        "multiplier": 0.90,
        "impact": "Temerrüt olasılığını %10 azalt",
        "direction": "reduce",
    },
    {
        "id": "stable_full_time_employment",
        "label": "Stabil Full-time İstihdam",
        "condition": "MonthsEmployed ≥ 36 ve Full-time",
        "multiplier": 0.90,
        "impact": "Temerrüt olasılığını %10 azalt",
        "direction": "reduce",
    },
    {
        "id": "cosigner_present",
        "label": "Kefil Var",
        "condition": "HasCoSigner = Yes",
        "multiplier": 0.90,
        "impact": "Temerrüt olasılığını %10 azalt",
        "direction": "reduce",
    },
    {
        "id": "high_dti",
        "label": "Yüksek DTI",
        "condition": "DTIRatio ≥ 0.50",
        "multiplier": 1.20,
        "impact": "Temerrüt olasılığını %20 artır",
        "direction": "increase",
    },
    {
        "id": "weak_credit_score",
        "label": "Zayıf Kredi Notu",
        "condition": "CreditScore < 600",
        "multiplier": 1.15,
        "impact": "Temerrüt olasılığını %15 artır",
        "direction": "increase",
    },
    {
        "id": "unstable_employment",
        "label": "Dengesiz İstihdam",
        "condition": "Unemployed veya MonthsEmployed < 6",
        "multiplier": 1.15,
        "impact": "Temerrüt olasılığını %15 artır",
        "direction": "increase",
    },
]

NAV_ITEMS = [
    {"key": "cockpit", "href": "/", "label": "Risk Kokpiti", "icon": "fa-shield-halved"},
    {"key": "overview", "href": "/genel-bakis", "label": "Genel Bakış", "icon": "fa-table-cells-large"},
    {"key": "scenarios", "href": "/demo-senaryolari", "label": "Demo Senaryoları", "icon": "fa-file-signature"},
    {"key": "monitoring", "href": "/model-izleme", "label": "Model İzleme", "icon": "fa-chart-simple"},
    {"key": "rules", "href": "/kurallar", "label": "Kurallar", "icon": "fa-sliders"},
]


def _page_context(request: Request, active: str, **extra: Any) -> dict[str, Any]:
    return {
        "request": request,
        "nav_items": NAV_ITEMS,
        "active": active,
        **extra,
    }


# --- Predict API ---------------------------------------------------------------------

class LoanApplication(BaseModel):
    Age: int = Field(..., ge=18)
    Income: float = Field(..., gt=0)
    LoanAmount: float = Field(..., ge=0)
    CreditScore: int
    MonthsEmployed: int = Field(..., ge=0)
    NumCreditLines: int = Field(..., ge=0)
    InterestRate: float = Field(..., ge=0)
    LoanTerm: int = Field(..., gt=0)
    DTIRatio: float | None = None
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str


def _model_dump(application: LoanApplication) -> dict[str, Any]:
    if hasattr(application, "model_dump"):
        return application.model_dump()
    return application.dict()


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def apply_business_rules(probability: float, application: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    """Configurable heuristic layer for reducing avoidable false positives."""
    adjusted = probability
    applied_rules: list[dict[str, Any]] = []

    def apply_rule(rule_id: str, condition: bool, multiplier: float, reason: str) -> None:
        nonlocal adjusted
        if not condition:
            return

        before = adjusted
        adjusted = _clamp_probability(adjusted * multiplier)
        applied_rules.append(
            {
                "id": rule_id,
                "multiplier": multiplier,
                "before": round(before * 100, 2),
                "after": round(adjusted * 100, 2),
                "reason": reason,
            }
        )

    credit_score = float(application.get("CreditScore", 0))
    dti_ratio = float(application.get("DTIRatio", 0))
    months_employed = float(application.get("MonthsEmployed", 0))
    employment_type = str(application.get("EmploymentType", ""))
    has_cosigner = str(application.get("HasCoSigner", "")).lower() == "yes"

    apply_rule(
        "strong_credit_score",
        credit_score >= 750,
        0.80,
        "CreditScore >= 750: reduce default probability by 20%.",
    )
    apply_rule(
        "low_dti",
        dti_ratio <= 0.35,
        0.90,
        "DTIRatio <= 0.35: reduce default probability by 10%.",
    )
    apply_rule(
        "stable_full_time_employment",
        months_employed >= 36 and employment_type == "Full-time",
        0.90,
        "Full-time employment for at least 36 months: reduce default probability by 10%.",
    )
    apply_rule(
        "cosigner_present",
        has_cosigner,
        0.90,
        "Co-signer present: reduce default probability by 10%.",
    )
    apply_rule(
        "high_dti",
        dti_ratio >= 0.50,
        1.20,
        "DTIRatio >= 0.50: increase default probability by 20%.",
    )
    apply_rule(
        "weak_credit_score",
        credit_score < 600,
        1.15,
        "CreditScore < 600: increase default probability by 15%.",
    )
    apply_rule(
        "unstable_employment",
        employment_type == "Unemployed" or months_employed < 6,
        1.15,
        "Unemployed or employed less than 6 months: increase default probability by 15%.",
    )

    return adjusted, applied_rules


# --- Page routes ---------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "cockpit.html",
        _page_context(
            request,
            active="cockpit",
            decision_threshold=round(float(decision_threshold), 3),
        ),
    )


@app.get("/genel-bakis", response_class=HTMLResponse)
def overview(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "overview.html",
        _page_context(
            request,
            active="overview",
            metrics=FINAL_METRICS,
            rule_count=len(BUSINESS_RULES),
        ),
    )


@app.get("/demo-senaryolari", response_class=HTMLResponse)
def demo_scenarios(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "scenarios.html",
        _page_context(
            request,
            active="scenarios",
            scenarios=DEMO_SCENARIOS,
        ),
    )


@app.get("/model-izleme", response_class=HTMLResponse)
def model_monitoring(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "monitoring.html",
        _page_context(
            request,
            active="monitoring",
            metrics=FINAL_METRICS,
            comparison=MODEL_COMPARISON,
            confusion=CONFUSION_MATRIX,
        ),
    )


@app.get("/kurallar", response_class=HTMLResponse)
def rules_page(request: Request) -> HTMLResponse:
    reduce_rules = [r for r in BUSINESS_RULES if r["direction"] == "reduce"]
    increase_rules = [r for r in BUSINESS_RULES if r["direction"] == "increase"]
    return templates.TemplateResponse(
        "rules.html",
        _page_context(
            request,
            active="rules",
            reduce_rules=reduce_rules,
            increase_rules=increase_rules,
            threshold=round(float(decision_threshold), 3),
            rule_version=BUSINESS_RULE_VERSION,
        ),
    )


@app.post("/predict")
def predict_risk(application: LoanApplication) -> dict[str, Any]:
    input_dict = _model_dump(application)
    engineered_df, _, x_scaled = prepare_inference_frame(input_dict, feature_names, scaler)
    engineered_dict = engineered_df.iloc[0].to_dict()

    model_probability = float(model.predict_proba(x_scaled)[0][1])
    adjusted_probability, applied_rules = apply_business_rules(model_probability, engineered_dict)
    final_prediction = int(adjusted_probability >= decision_threshold)

    return {
        "risk_durumu": final_prediction,
        "temerrut_olasiligi": round(adjusted_probability * 100, 2),
        "model_risk_durumu": int(model_probability >= decision_threshold),
        "model_temerrut_olasiligi": round(model_probability * 100, 2),
        "hesaplanan_dti": round(float(engineered_dict["DTIRatio"]), 4),
        "business_rule_version": BUSINESS_RULE_VERSION,
        "business_rule_adjustments": applied_rules,
        "decision_threshold": round(float(decision_threshold), 3),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
