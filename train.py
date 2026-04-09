"""
train.py
--------
Train and evaluate multiple classifiers on the churn dataset.
Saves the best model to models/best_model.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE

from preprocess import load_data, clean_data, encode_features, split_and_scale


def apply_smote(X_train, y_train):
    """Handle class imbalance using SMOTE oversampling."""
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE — Class distribution: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


def get_models():
    """Return a dict of models to train and compare."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }


def evaluate_model(name, model, X_test, y_test):
    """Print evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    return {
        "model": model,
        "name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def train(data_path: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    # Load & preprocess
    df = load_data(data_path)
    df = clean_data(df)
    df, _ = encode_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    # Handle class imbalance
    X_train, y_train = apply_smote(X_train, y_train)

    # Train all models
    models = get_models()
    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        result = evaluate_model(name, model, X_test, y_test)
        results.append(result)

    # Pick best model by ROC-AUC
    best = max(results, key=lambda x: x["roc_auc"])
    print(f"\n✅ Best Model: {best['name']} (ROC-AUC: {best['roc_auc']:.4f})")

    # Save best model and scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(best["model"], "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("💾 Saved best model to models/best_model.pkl")


if __name__ == "__main__":
    train()
