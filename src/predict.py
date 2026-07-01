"""
predict.py
Utility for predicting churn probability for a single customer.
Can be imported by the Streamlit app or called from command line.

Usage:
  python src/predict.py
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')

GEO_MAP = {'France': 0, 'Spain': 1, 'Germany': 2}
GEN_MAP = {'Female': 0, 'Male': 1}


def load_model():
    """Load the saved model bundle from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run run_pipeline.py first to train and save the model."
        )
    bundle = joblib.load(MODEL_PATH)
    return bundle['model'], bundle['scaler'], bundle['features']


def build_features(customer: dict) -> pd.DataFrame:
    """
    Given a raw customer dict (with original fields), compute all
    engineered features and return a single-row DataFrame ready for prediction.
    """
    row = pd.Series(customer)

    balance = float(row.get('Balance', 0))
    salary = float(row.get('EstimatedSalary', 1))
    tenure = float(row.get('Tenure', 0))
    products = float(row.get('NumOfProducts', 1))
    active = float(row.get('IsActiveMember', 0))
    cc = float(row.get('HasCrCard', 0))
    credit_score = float(row.get('CreditScore', 650))

    features = {
        'CreditScore': credit_score,
        'Age': float(row.get('Age', 40)),
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': products,
        'HasCrCard': cc,
        'IsActiveMember': active,
        'EstimatedSalary': salary,
        'BalanceSalaryRatio': balance / (salary + 1),
        'ProductsPerTenure': products / (tenure + 1),
        'EngagementScore': 0.7 * active + 0.3 * cc,
        'WealthScore': (balance / 250898 + salary / 199992 + credit_score / 850),
        'RelationshipStrength': (
            0.4 * (0.7 * active + 0.3 * cc) +
            0.3 * ((products - 1) / 3) +
            0.3 * min(tenure / 10, 1)
        ),
        'Geography_enc': GEO_MAP.get(row.get('Geography', 'France'), 0),
        'Gender_enc': GEN_MAP.get(row.get('Gender', 'Male'), 0),
    }
    return pd.DataFrame([features])


def predict_churn(customer: dict):
    """
    Predict churn probability and return full result dict.

    Args:
        customer: dict with keys matching the European Bank dataset columns.

    Returns:
        dict with keys: churn_probability (%), risk_level, rsi, recommendation.
    """
    model, scaler, feature_cols = load_model()

    X = build_features(customer)[feature_cols]
    X_sc = scaler.transform(X)
    prob = float(model.predict_proba(X_sc)[0, 1])

    # RSI
    active = float(customer.get('IsActiveMember', 0))
    cc = float(customer.get('HasCrCard', 0))
    tenure = float(customer.get('Tenure', 0))
    products = float(customer.get('NumOfProducts', 1))
    prod_map = {1: 0.25, 2: 1.0, 3: 0.20, 4: 0.0}
    prod_depth = prod_map.get(int(products), 0.25)
    engagement = 0.7 * active + 0.3 * cc
    loyalty = min(tenure / 10, 1)
    rsi = round((0.4 * engagement + 0.3 * prod_depth + 0.3 * loyalty) * 100, 1)

    if rsi <= 30:
        rsi_cat = 'High Risk'
    elif rsi <= 60:
        rsi_cat = 'Moderate Risk'
    elif rsi <= 80:
        rsi_cat = 'Stable'
    else:
        rsi_cat = 'Loyal'

    # Risk level
    if prob > 0.80:
        risk_level = 'High Risk'
    elif prob > 0.50:
        risk_level = 'Moderate Risk'
    else:
        risk_level = 'Low Risk'

    # Recommendation
    balance = float(customer.get('Balance', 0))
    if prob > 0.80:
        rec = 'Immediate Outreach'
    elif active == 0:
        rec = 'Reactivation Campaign'
    elif products == 1:
        rec = 'Cross-Sell Programme'
    elif balance > 97199:
        rec = 'Relationship Manager Assignment'
    else:
        rec = 'Standard Retention Programme'

    return {
        'churn_probability': round(prob * 100, 1),
        'risk_level': risk_level,
        'rsi': rsi,
        'rsi_category': rsi_cat,
        'recommendation': rec
    }


if __name__ == '__main__':
    # Demo prediction
    example_customer = {
        'CreditScore': 580,
        'Geography': 'Germany',
        'Gender': 'Female',
        'Age': 52,
        'Tenure': 2,
        'Balance': 145000,
        'NumOfProducts': 1,
        'HasCrCard': 1,
        'IsActiveMember': 0,
        'EstimatedSalary': 98000
    }
    result = predict_churn(example_customer)
    print("\nExample prediction:")
    print(f"  Churn probability : {result['churn_probability']}%")
    print(f"  Risk level        : {result['risk_level']}")
    print(f"  RSI               : {result['rsi']} ({result['rsi_category']})")
    print(f"  Recommendation    : {result['recommendation']}")
