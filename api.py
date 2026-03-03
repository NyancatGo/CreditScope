from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="CreditScope - Risk Analizi API")

# CORS Settings (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load machine learning models at startup to save processing time
print("Modeller belleğe yükleniyor...")
model = joblib.load("xgboost_optimized.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Pydantic BaseModel for Input Validation
class LoanApplication(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: int
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str

@app.post("/predict")
def predict_risk(application: LoanApplication):
    # Convert input data to pandas DataFrame
    input_dict = application.model_dump()
    df = pd.DataFrame([input_dict])

    # Categorical Columns for One-Hot Encoding
    categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

    # Apply One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Ensure all feature names are present, fill with 0 (since user input will lack many categories)
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Ensure column order matches exactly what the model saw during training
    df_encoded = df_encoded[feature_names]

    # Standardize data using the loaded Scaler
    X_scaled = scaler.transform(df_encoded)

    # Prediction
    prediction = int(model.predict(X_scaled)[0])
    prediction_proba = float(model.predict_proba(X_scaled)[0][1])  # Default probability (class 1)
    
    return {
        "risk_durumu": prediction,
        "temerrut_olasiligi": round(prediction_proba * 100, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
