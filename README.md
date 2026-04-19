# CreditScope

CreditScope is a compact credit default risk analysis project. It trains an XGBoost model, serves predictions with FastAPI, and provides a simple web UI for credit application scoring.

The project is recall-oriented: missing a truly risky customer is treated as more costly than sending an extra application to manual review.

## Project Structure

```text
CreditScope/
|-- api.py                    # FastAPI app, UI serving, business rules
|-- preprocessing.py          # Shared training and inference preprocessing
|-- training.py               # SMOTE + optimized XGBoost training
|-- shap_analysis.py          # FP/FN error analysis and SHAP outputs
|-- Loan_default.csv          # Source dataset
|-- xgboost_optimized.pkl     # Active model used by the API
|-- scaler.pkl                # Active scaler used by the API
|-- feature_names.pkl         # Active model feature order
|-- decision_threshold.pkl    # Recall-oriented probability threshold
|-- requirements.txt          # Python dependencies
|-- static/
|   |-- style.css
|   `-- script.js
`-- templates/
    `-- index.html
```

Generated reports and plots are written to `outputs/` when training or SHAP analysis runs. That folder is intentionally ignored by git.

## Model Logic

Shared feature engineering in `preprocessing.py` creates:

- `DTIRatio = LoanAmount / Income`
- `Age_Income_Interaction = Age * Income`

The API recalculates `DTIRatio` from raw `LoanAmount` and `Income`, so the frontend does not need to be trusted as the source of truth.

## Business Rules

`api.py` applies a transparent post-model rule layer:

| Rule | Adjustment |
| --- | --- |
| `CreditScore >= 750` | Reduce risk by 20% |
| `DTIRatio <= 0.35` | Reduce risk by 10% |
| Full-time employment for at least 36 months | Reduce risk by 10% |
| Co-signer present | Reduce risk by 10% |
| `DTIRatio >= 0.50` | Increase risk by 20% |
| `CreditScore < 600` | Increase risk by 15% |
| Unemployed or employed less than 6 months | Increase risk by 15% |

The `/predict` response returns the raw model probability, adjusted probability, calculated DTI, decision threshold, and the applied rule list.

## Setup

```bash
py -m pip install -r requirements.txt
```

## Train

```bash
py training.py
```

This refreshes:

- `xgboost_optimized.pkl`
- `scaler.pkl`
- `feature_names.pkl`
- `decision_threshold.pkl`

## Explainability

```bash
py shap_analysis.py
```

This creates:

- `outputs/shap/shap_summary_False_Positives.png`
- `outputs/shap/shap_summary_False_Negatives.png`
- `outputs/shap/error_analysis_feature_means.csv`

## Run

```bash
py -m uvicorn api:app --reload
```

Open:

- UI: http://127.0.0.1:8000/
- API docs: http://127.0.0.1:8000/docs
