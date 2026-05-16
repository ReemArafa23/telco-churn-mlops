import sys
sys.path.insert(0, '.')
from app import predict, CustomerFeatures
import json

# Test with a sample customer
test_customer = CustomerFeatures(
    customerID="test-001",
    gender="Female",
    SeniorCitizen=0,
    Partner="Yes",
    Dependents="No",
    tenure=12,
    PhoneService="Yes",
    MultipleLines="No",
    InternetService="DSL",
    OnlineSecurity="No",
    OnlineBackup="Yes",
    DeviceProtection="No",
    TechSupport="No",
    StreamingTV="No",
    StreamingMovies="No",
    Contract="Month-to-month",
    PaperlessBilling="Yes",
    PaymentMethod="Electronic check",
    MonthlyCharges=29.85,
    TotalCharges=346.45
)

# Test prediction
result = predict([test_customer])
print("✓ Prediction successful!")
print(json.dumps(result, indent=2))
