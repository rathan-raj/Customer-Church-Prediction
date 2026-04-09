"""
predict.py
----------
Load the saved model and make predictions on new customer data.
"""

import joblib
import numpy as np
import pandas as pd

from preprocess import clean_data, encode_features


def load_model(model_path: str = "../models/best_model.pkl",
               scaler_path: str = "../models/scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model and scaler loaded.")
    return model, scaler


def predict_churn(customer_data: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """
    Given a DataFrame of new customer records, return churn predictions.
    customer_data should have the same columns as training data (minus customerID and Churn).
    """
    # Preprocess
    df, _ = encode_features(customer_data)
    X_scaled = scaler.transform(df)

    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    result = customer_data.copy()
    result["Churn_Prediction"] = predictions
    result["Churn_Probability"] = probabilities.round(3)
    result["Risk_Level"] = pd.cut(
        probabilities,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    )

    return result


if __name__ == "__main__":
    # Example usage with a sample customer
    sample = pd.DataFrame([{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35,
        "TotalCharges": 351.75
    }])

    model, scaler = load_model()
    results = predict_churn(sample, model, scaler)
    print(results[["Churn_Prediction", "Churn_Probability", "Risk_Level"]])
