from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

import joblib
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from auth import (
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE_SECONDS,
    authenticate_user,
    create_first_user,
    create_session,
    delete_session,
    get_user_by_session_token,
    has_users,
)
from preprocessing import ROOT_DIR, prepare_inference_frame


MODEL_PATH = ROOT_DIR / "xgboost_optimized.pkl"
SCALER_PATH = ROOT_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = ROOT_DIR / "feature_names.pkl"
DECISION_THRESHOLD_PATH = ROOT_DIR / "decision_threshold.pkl"
TEMPLATES_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"
DATA_DIR = ROOT_DIR / "data"
VALIDATION_SNAPSHOT_PATH = DATA_DIR / "final_validation_snapshot.json"
BUSINESS_RULE_VERSION = "default-review-2026-04-19"
ASSET_VERSION = "20260420-week13"


app = FastAPI(
    title="CreditScope - Risk Analizi API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

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


# --- Final validation snapshot -------------------------------------------------------

def load_validation_snapshot() -> dict[str, Any]:
    """Load tracked final evidence used by Week 12/13 UI pages."""
    if not VALIDATION_SNAPSHOT_PATH.exists():
        raise FileNotFoundError(f"Final validation snapshot not found: {VALIDATION_SNAPSHOT_PATH}")
    return json.loads(VALIDATION_SNAPSHOT_PATH.read_text(encoding="utf-8"))


VALIDATION_SNAPSHOT = load_validation_snapshot()
DEMO_SCENARIOS = VALIDATION_SNAPSHOT["demo_scenarios"]
FINAL_METRICS = VALIDATION_SNAPSHOT["final_metrics"]
MODEL_COMPARISON = VALIDATION_SNAPSHOT["model_comparison"]
CONFUSION_MATRIX = VALIDATION_SNAPSHOT["confusion_matrix"]


# --- Business rules ------------------------------------------------------------------

def _rule_float(application: dict[str, Any], key: str) -> float:
    return float(application.get(key, 0))


def _rule_text(application: dict[str, Any], key: str) -> str:
    return str(application.get(key, ""))


BUSINESS_RULES = [
    {
        "id": "strong_credit_score",
        "label": "Güçlü Kredi Notu",
        "condition": "CreditScore >= 750",
        "multiplier": 0.80,
        "impact": "Temerrüt olasılığını %20 azalt",
        "direction": "reduce",
        "reason": "CreditScore >= 750: reduce default probability by 20%.",
        "predicate": lambda app: _rule_float(app, "CreditScore") >= 750,
    },
    {
        "id": "low_dti",
        "label": "Düşük DTI",
        "condition": "DTIRatio <= 0.35",
        "multiplier": 0.90,
        "impact": "Temerrüt olasılığını %10 azalt",
        "direction": "reduce",
        "reason": "DTIRatio <= 0.35: reduce default probability by 10%.",
        "predicate": lambda app: _rule_float(app, "DTIRatio") <= 0.35,
    },
    {
        "id": "stable_full_time_employment",
        "label": "Stabil Full-time İstihdam",
        "condition": "MonthsEmployed >= 36 ve Full-time",
        "multiplier": 0.90,
        "impact": "Temerrüt olasılığını %10 azalt",
        "direction": "reduce",
        "reason": "Full-time employment for at least 36 months: reduce default probability by 10%.",
        "predicate": lambda app: _rule_float(app, "MonthsEmployed") >= 36
        and _rule_text(app, "EmploymentType") == "Full-time",
    },
    {
        "id": "cosigner_present",
        "label": "Kefil Var",
        "condition": "HasCoSigner = Yes",
        "multiplier": 0.90,
        "impact": "Temerrüt olasılığını %10 azalt",
        "direction": "reduce",
        "reason": "Co-signer present: reduce default probability by 10%.",
        "predicate": lambda app: _rule_text(app, "HasCoSigner").lower() == "yes",
    },
    {
        "id": "high_dti",
        "label": "Yüksek DTI",
        "condition": "DTIRatio >= 0.50",
        "multiplier": 1.20,
        "impact": "Temerrüt olasılığını %20 artır",
        "direction": "increase",
        "reason": "DTIRatio >= 0.50: increase default probability by 20%.",
        "predicate": lambda app: _rule_float(app, "DTIRatio") >= 0.50,
    },
    {
        "id": "weak_credit_score",
        "label": "Zayıf Kredi Notu",
        "condition": "CreditScore < 600",
        "multiplier": 1.15,
        "impact": "Temerrüt olasılığını %15 artır",
        "direction": "increase",
        "reason": "CreditScore < 600: increase default probability by 15%.",
        "predicate": lambda app: _rule_float(app, "CreditScore") < 600,
    },
    {
        "id": "unstable_employment",
        "label": "Dengesiz İstihdam",
        "condition": "Unemployed veya MonthsEmployed < 6",
        "multiplier": 1.15,
        "impact": "Temerrüt olasılığını %15 artır",
        "direction": "increase",
        "reason": "Unemployed or employed less than 6 months: increase default probability by 15%.",
        "predicate": lambda app: _rule_text(app, "EmploymentType") == "Unemployed"
        or _rule_float(app, "MonthsEmployed") < 6,
    },
]


NAV_ITEMS = [
    {"key": "cockpit", "href": "/", "label": "Risk Kokpiti", "icon": "fa-shield-halved"},
    {"key": "overview", "href": "/genel-bakis", "label": "Genel Bakış", "icon": "fa-table-cells-large"},
    {"key": "scenarios", "href": "/demo-senaryolari", "label": "Demo Senaryoları", "icon": "fa-file-signature"},
    {"key": "monitoring", "href": "/model-izleme", "label": "Model İzleme", "icon": "fa-chart-simple"},
    {"key": "rules", "href": "/kurallar", "label": "Kurallar", "icon": "fa-sliders"},
]


def _current_user(request: Request) -> dict[str, Any] | None:
    return get_user_by_session_token(request.cookies.get(SESSION_COOKIE_NAME))


def _safe_next_path(next_path: str | None) -> str:
    if not next_path or not next_path.startswith("/") or next_path.startswith("//"):
        return "/"
    return next_path


def _login_redirect(request: Request) -> RedirectResponse:
    target = str(request.url.path)
    if request.url.query:
        target = f"{target}?{request.url.query}"
    return RedirectResponse(url=f"/login?next={quote(target)}", status_code=303)


def _page_context(request: Request, active: str, **extra: Any) -> dict[str, Any]:
    current_user = extra.pop("current_user", None)
    if current_user is None:
        current_user = _current_user(request)
    return {
        "request": request,
        "nav_items": NAV_ITEMS,
        "active": active,
        "current_user": current_user,
        **extra,
    }


def _protected_template(
    request: Request,
    template_name: str,
    active: str,
    **extra: Any,
) -> HTMLResponse | RedirectResponse:
    current_user = _current_user(request)
    if current_user is None:
        return _login_redirect(request)
    return templates.TemplateResponse(
        template_name,
        _page_context(request, active=active, current_user=current_user, **extra),
    )


def require_api_user(request: Request) -> dict[str, Any]:
    current_user = _current_user(request)
    if current_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return current_user


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
    """Apply the displayable rule definitions used by the rules page."""
    adjusted = probability
    applied_rules: list[dict[str, Any]] = []

    for rule in BUSINESS_RULES:
        predicate = rule["predicate"]
        if not predicate(application):
            continue

        before = adjusted
        multiplier = float(rule["multiplier"])
        adjusted = _clamp_probability(adjusted * multiplier)
        applied_rules.append(
            {
                "id": rule["id"],
                "multiplier": multiplier,
                "before": round(before * 100, 2),
                "after": round(adjusted * 100, 2),
                "reason": rule["reason"],
            }
        )

    return adjusted, applied_rules


# --- Page routes ---------------------------------------------------------------------

@app.get("/login", response_class=HTMLResponse, response_model=None)
def login_page(request: Request) -> HTMLResponse | RedirectResponse:
    next_path = _safe_next_path(request.query_params.get("next"))
    if _current_user(request) is not None:
        return RedirectResponse(url=next_path, status_code=303)

    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "mode": "login",
            "next_path": next_path,
            "error": None,
            "registered": request.query_params.get("registered") == "1",
            "registration_available": not has_users(),
        },
    )


@app.post("/login", response_class=HTMLResponse, response_model=None)
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next_path: str = Form("/"),
) -> HTMLResponse | RedirectResponse:
    next_path = _safe_next_path(next_path)
    user = authenticate_user(username, password)
    if user is None:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "mode": "login",
                "next_path": next_path,
                "error": "Kullanıcı adı veya şifre hatalı.",
                "registered": False,
                "registration_available": not has_users(),
            },
            status_code=401,
        )

    session_token = create_session(int(user["id"]))
    response = RedirectResponse(url=next_path, status_code=303)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=False,
    )
    return response


@app.get("/logout")
def logout(request: Request) -> RedirectResponse:
    delete_session(request.cookies.get(SESSION_COOKIE_NAME))
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response


@app.get("/register", response_class=HTMLResponse, response_model=None)
def register_page(request: Request) -> HTMLResponse:
    if has_users():
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "mode": "register_disabled",
                "next_path": "/",
                "error": "İlk admin kullanıcısı zaten oluşturulmuş. Kayıt devre dışı.",
                "registered": False,
                "registration_available": False,
            },
            status_code=403,
        )

    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "mode": "register",
            "next_path": "/",
            "error": None,
            "registered": False,
            "registration_available": True,
        },
    )


@app.post("/register", response_class=HTMLResponse, response_model=None)
def register_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
) -> HTMLResponse | RedirectResponse:
    if has_users():
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "mode": "register_disabled",
                "next_path": "/",
                "error": "İlk admin kullanıcısı zaten oluşturulmuş. Kayıt devre dışı.",
                "registered": False,
                "registration_available": False,
            },
            status_code=403,
        )

    try:
        create_first_user(username, password)
    except (PermissionError, ValueError) as exc:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "mode": "register",
                "next_path": "/",
                "error": str(exc),
                "registered": False,
                "registration_available": True,
            },
            status_code=400,
        )

    return RedirectResponse(url="/login?registered=1", status_code=303)


@app.get("/docs", response_class=HTMLResponse, response_model=None)
def protected_docs(request: Request) -> HTMLResponse | RedirectResponse:
    if _current_user(request) is None:
        return _login_redirect(request)
    return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{app.title} - Docs")


@app.get("/openapi.json")
def protected_openapi(_: dict[str, Any] = Depends(require_api_user)) -> JSONResponse:
    return JSONResponse(app.openapi())


@app.get("/", response_class=HTMLResponse, response_model=None)
def index(request: Request) -> HTMLResponse | RedirectResponse:
    return _protected_template(
        request,
        "cockpit.html",
        active="cockpit",
        decision_threshold=round(float(decision_threshold), 3),
    )


@app.get("/genel-bakis", response_class=HTMLResponse, response_model=None)
def overview(request: Request) -> HTMLResponse | RedirectResponse:
    return _protected_template(
        request,
        "overview.html",
        active="overview",
        metrics=FINAL_METRICS,
        rule_count=len(BUSINESS_RULES),
    )


@app.get("/demo-senaryolari", response_class=HTMLResponse, response_model=None)
def demo_scenarios(request: Request) -> HTMLResponse | RedirectResponse:
    return _protected_template(
        request,
        "scenarios.html",
        active="scenarios",
        scenarios=DEMO_SCENARIOS,
    )


@app.get("/model-izleme", response_class=HTMLResponse, response_model=None)
def model_monitoring(request: Request) -> HTMLResponse | RedirectResponse:
    return _protected_template(
        request,
        "monitoring.html",
        active="monitoring",
        metrics=FINAL_METRICS,
        comparison=MODEL_COMPARISON,
        confusion=CONFUSION_MATRIX,
    )


@app.get("/kurallar", response_class=HTMLResponse, response_model=None)
def rules_page(request: Request) -> HTMLResponse | RedirectResponse:
    reduce_rules = [r for r in BUSINESS_RULES if r["direction"] == "reduce"]
    increase_rules = [r for r in BUSINESS_RULES if r["direction"] == "increase"]
    return _protected_template(
        request,
        "rules.html",
        active="rules",
        reduce_rules=reduce_rules,
        increase_rules=increase_rules,
        threshold=round(float(decision_threshold), 3),
        rule_version=BUSINESS_RULE_VERSION,
    )


@app.post("/predict")
def predict_risk(
    application: LoanApplication,
    _: dict[str, Any] = Depends(require_api_user),
) -> dict[str, Any]:
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
