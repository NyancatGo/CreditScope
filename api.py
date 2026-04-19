from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from preprocessing import ROOT_DIR, prepare_inference_frame


MODEL_PATH = ROOT_DIR / "xgboost_optimized.pkl"
SCALER_PATH = ROOT_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = ROOT_DIR / "feature_names.pkl"
DECISION_THRESHOLD_PATH = ROOT_DIR / "decision_threshold.pkl"
TEMPLATES_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"
BUSINESS_RULE_VERSION = "default-review-2026-04-19"


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

print("Loading CreditScope model artifacts...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)
decision_threshold = joblib.load(DECISION_THRESHOLD_PATH) if DECISION_THRESHOLD_PATH.exists() else 0.50


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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(TEMPLATES_DIR / "index.html")


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
