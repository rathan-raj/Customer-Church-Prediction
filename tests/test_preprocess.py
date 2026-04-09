"""
Basic unit tests for preprocess.py
"""

import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import clean_data, encode_features, split_and_scale


def _sample_df():
    """Return a minimal churn dataframe for testing.

    Needs at least 2 rows per Churn class so stratified train/test split works.
    """
    return pd.DataFrame({
        "customerID": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        "gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        "Partner": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
        "Dependents": ["No", "No", "Yes", "No", "No", "Yes", "No", "No", "Yes", "No"],
        "tenure": [1, 12, 24, 36, 5, 48, 2, 60, 8, 18],
        "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes"],
        "MultipleLines": ["No", "Yes", "No phone service", "No", "Yes", "Yes", "No phone service", "No", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "No", "Fiber optic", "DSL", "Fiber optic", "No", "DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes", "No", "No internet service", "No", "Yes", "No", "No internet service", "Yes", "No"],
        "OnlineBackup": ["Yes", "No", "Yes", "No internet service", "No", "Yes", "No", "No internet service", "No", "Yes"],
        "DeviceProtection": ["No", "Yes", "No", "No internet service", "No", "No", "Yes", "No internet service", "No", "Yes"],
        "TechSupport": ["No", "No", "No", "No internet service", "No", "Yes", "No", "No internet service", "No", "No"],
        "StreamingTV": ["No", "Yes", "No", "No internet service", "Yes", "No", "Yes", "No internet service", "No", "Yes"],
        "StreamingMovies": ["No", "No", "No", "No internet service", "Yes", "No", "No", "No internet service", "Yes", "No"],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "Month-to-month",
                     "One year", "Month-to-month", "Two year", "Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No", "No", "Yes", "Yes", "No", "Yes", "No", "Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                          "Credit card (automatic)", "Electronic check", "Mailed check",
                          "Electronic check", "Bank transfer (automatic)", "Credit card (automatic)", "Mailed check"],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70, 49.95, 85.10, 20.05, 65.60, 55.00],
        "TotalCharges": ["29.85", "680.35", "1397.47", " ", "151.65", "2397.55", "170.20", "1203.00", "524.80", "990.00"],
        "Churn": ["No", "No", "No", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
    })


def test_clean_data_drops_customer_id():
    df = clean_data(_sample_df())
    assert "customerID" not in df.columns


def test_clean_data_churn_binary():
    df = clean_data(_sample_df())
    assert set(df["Churn"].unique()).issubset({0, 1})


def test_clean_data_total_charges_numeric():
    df = clean_data(_sample_df())
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert df["TotalCharges"].isna().sum() == 0


def test_encode_features_no_object_columns():
    df = clean_data(_sample_df())
    df_encoded, cat_cols = encode_features(df)
    assert len(df_encoded.select_dtypes(include="object").columns) == 0


def test_split_and_scale_shapes():
    df = clean_data(_sample_df())
    df_encoded, _ = encode_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_encoded, test_size=0.2)
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] == X_test.shape[1]
