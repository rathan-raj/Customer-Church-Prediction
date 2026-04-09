"""
preprocess.py
-------------
Functions to clean and prepare the Telco Churn dataset for modelling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Fix TotalCharges column (contains whitespace strings)
    - Drop customerID (not useful for prediction)
    - Convert target to binary (0/1)
    """
    df = df.copy()

    # Fix TotalCharges — some entries are blank strings
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop customer ID
    df.drop(columns=["customerID"], inplace=True)

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    print(f"Churn rate: {df['Churn'].mean():.2%}")
    return df


def encode_features(df: pd.DataFrame):
    """
    Encode categorical columns using Label Encoding.
    Returns the encoded dataframe and the list of feature columns.
    """
    df = df.copy()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    print(f"Encoded {len(categorical_cols)} categorical columns.")
    return df, categorical_cols


def split_and_scale(df: pd.DataFrame, target: str = "Churn", test_size: float = 0.2):
    """
    Split into train/test sets and apply StandardScaler to numeric features.
    Returns: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
