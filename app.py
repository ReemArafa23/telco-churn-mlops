from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import joblib
import pandas as pd
import numpy as np
import sys
import psutil
from datetime import datetime
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

# Helper used when pipeline was serialized
def map_yes_no(values):
    array = np.asarray(values)
    return (array == 'Yes').astype(int)

# Ensure function is available for joblib unpickling by adding to __main__
import __main__
if not hasattr(__main__, 'map_yes_no'):
    __main__.map_yes_no = map_yes_no

# Load pipeline and reference data at startup
PIPELINE_PATH = "churn_inference_pipeline.joblib"
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

try:
    pipeline = joblib.load(PIPELINE_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load pipeline from {PIPELINE_PATH}: {e}")

# Compute high-value threshold from available dataset to mirror Phase 2 segmentation
try:
    _df_ref = pd.read_csv(DATA_PATH)
    _df_ref['TotalCharges'] = pd.to_numeric(_df_ref['TotalCharges'], errors='coerce').fillna(0)
    HIGH_VALUE_TOTALCHARGES = float(_df_ref['TotalCharges'].quantile(0.75))
except Exception:
    HIGH_VALUE_TOTALCHARGES = 100.0  # safe default

# Business segment mapping (same text used in notebook)
_DESCRIPTIONS = {
    'High-Value Flight Risk': 'High predicted churn and high lifetime value (top 25% TotalCharges).',
    'New-Customer Flight Risk': 'Recently acquired customers with high predicted churn (tenure <= 6 months).',
    'Mid-Risk Retention Opportunity': 'Moderate predicted churn — good candidates for targeted offers.',
    'Low Risk': 'Low predicted churn; standard engagement.'
}
_ACTIONS = {
    'High-Value Flight Risk': 'Immediate retention: VIP outreach, custom discounts, account review, loyalty incentives.',
    'New-Customer Flight Risk': 'Onboarding improvements: targeted welcome offers, quick-check calls, trial extensions.',
    'Mid-Risk Retention Opportunity': 'Nudge campaigns: time-limited promotions, bundle suggestions, service checks.',
    'Low Risk': 'Standard lifecycle marketing.'
}

app = FastAPI(title="Telco Churn Inference API", version="1.0")

# Prometheus metrics for model-level monitoring
churn_predictions_counter = Counter(
    'churn_predictions_total',
    'Total number of churn predictions',
    ['prediction_class']
)

prediction_requests_counter = Counter(
    'prediction_requests_total',
    'Total number of prediction requests'
)

prediction_errors_counter = Counter(
    'prediction_errors_total',
    'Total number of prediction errors'
)

cpu_usage_gauge = Gauge(
    'cpu_usage_percent',
    'Current CPU usage percentage'
)

memory_usage_gauge = Gauge(
    'memory_usage_percent',
    'Current memory usage percentage'
)

# Integrate Prometheus FastAPI Instrumentator for automatic HTTP metrics
Instrumentator().instrument(app).expose(app)

# Pydantic model capturing Telco features used in pipeline
class CustomerFeatures(BaseModel):
    customerID: Optional[str]
    gender: Optional[str]
    SeniorCitizen: Optional[int] = Field(None, ge=0, le=1)
    Partner: Optional[str]
    Dependents: Optional[str]
    tenure: Optional[int]
    PhoneService: Optional[str]
    MultipleLines: Optional[str]
    InternetService: Optional[str]
    OnlineSecurity: Optional[str]
    OnlineBackup: Optional[str]
    DeviceProtection: Optional[str]
    TechSupport: Optional[str]
    StreamingTV: Optional[str]
    StreamingMovies: Optional[str]
    Contract: Optional[str]
    PaperlessBilling: Optional[str]
    PaymentMethod: Optional[str]
    MonthlyCharges: Optional[float]
    TotalCharges: Optional[float]

    class Config:
        json_schema_extra = {
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 346.45,
            }
        }

@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint with system metrics."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_info = psutil.virtual_memory()
    
    # Update Prometheus gauges
    cpu_usage_gauge.set(cpu_percent)
    memory_usage_gauge.set(memory_info.percent)
    
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "cpu_percent": cpu_percent,
        "memory_percent": memory_info.percent,
        "memory_available_mb": memory_info.available / (1024 ** 2),
    }

@app.post("/predict")
def predict(customers: List[CustomerFeatures]):
    """Prediction endpoint with model-level metrics tracking."""
    if not customers:
        raise HTTPException(status_code=400, detail="Empty request body")

    prediction_requests_counter.inc()

    # Convert to DataFrame
    try:
        data = pd.DataFrame([c.dict() for c in customers])
    except Exception as e:
        prediction_errors_counter.inc()
        raise HTTPException(status_code=400, detail=f"Invalid input format: {e}")

    # Ensure numeric columns
    if 'TotalCharges' in data.columns:
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)
    else:
        data['TotalCharges'] = 0.0

    if 'MonthlyCharges' in data.columns:
        data['MonthlyCharges'] = pd.to_numeric(data['MonthlyCharges'], errors='coerce').fillna(0)
    else:
        data['MonthlyCharges'] = 0.0

    if 'tenure' in data.columns:
        data['tenure'] = pd.to_numeric(data['tenure'], errors='coerce').fillna(0).astype(int)
    else:
        data['tenure'] = 0

    # Run inference
    try:
        if hasattr(pipeline, 'predict_proba'):
            proba = pipeline.predict_proba(data)[:, 1]
        else:
            # fallback to predict if probabilities not available
            proba = pipeline.predict(data)
    except Exception as e:
        prediction_errors_counter.inc()
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # Create segmentation using same business rules as Phase 2
    df_out = data.copy()
    df_out['_churn_proba'] = proba
    high_value_thresh = HIGH_VALUE_TOTALCHARGES

    segments = []
    churn_count = 0
    no_churn_count = 0
    
    for _, row in df_out.iterrows():
        p = float(row['_churn_proba'])
        total = float(row.get('TotalCharges', 0.0))
        tenure = int(row.get('tenure', 0))
        if (p >= 0.6) and (total >= high_value_thresh):
            seg = 'High-Value Flight Risk'
            churn_count += 1
        elif (p >= 0.6) and (tenure <= 6):
            seg = 'New-Customer Flight Risk'
            churn_count += 1
        elif (p >= 0.4) and (p < 0.6):
            seg = 'Mid-Risk Retention Opportunity'
            churn_count += 1
        else:
            seg = 'Low Risk'
            no_churn_count += 1
        segments.append(seg)

    # Update prediction counters
    churn_predictions_counter.labels(prediction_class='churn').inc(churn_count)
    churn_predictions_counter.labels(prediction_class='no_churn').inc(no_churn_count)

    # Build response
    results = []
    for idx, cust in enumerate(customers):
        seg = segments[idx]
        results.append({
            'customerID': cust.customerID,
            'churn_proba': float(round(proba[idx], 6)),
            'segment': seg,
            'characteristics': _DESCRIPTIONS.get(seg),
            'recommended_action': _ACTIONS.get(seg),
        })

    return {"predictions": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=1)
