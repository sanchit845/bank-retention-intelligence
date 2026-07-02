"""
inference.py
Single home for ad-hoc and programmatic churn prediction.

Consolidates the previously separate src/predict.py (CLI/demo path) and
src/utils.predict_single (dashboard path). Bodies preserved verbatim; only
the magic-number 97199 has been promoted to a named constant — value unchanged.
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR,
    DEFAULT_MODEL_NAME,
    HIGH_VALUE_BALANCE_THRESHOLD,
    CHURN_HIGH_THRESHOLD,
    CHURN_MODERATE_THRESHOLD,
    RANDOM_STATE,
)


# ── Encoding maps (kept module-local; same shape as before) ───────────────
GEO_MAP = {'France': 0, 'Spain': 1, 'Germany': 2}
GEN_MAP = {'Female': 0, 'Male': 1}


# ── Recommendation rule (was utils.predict_single inline + clustering.get_recommendation) ──
def _recommend(row) -> str:
    """Priority-ordered retention rules. First match wins."""
    prob     = row.get('ChurnProbability', 0)
    active   = row['IsActiveMember']
    products = row['NumOfProducts']
    balance  = row['Balance']

    if prob > CHURN_HIGH_THRESHOLD:
        return 'Immediate Outreach'
    elif active == 0:
        return 'Reactivation Campaign'
    elif products == 1:
        return 'Cross-Sell Programme'
    elif balance > HIGH_VALUE_BALANCE_THRESHOLD:
        return 'Relationship Manager Assignment'
    else:
        return 'Standard Retention Programme'


# ── RSI (for single-customer scoring) ────────────────────────────────────
_PROD_MAP = {1: 0.25, 2: 1.0, 3: 0.20, 4: 0.0}


def _rsi_for(active: float, cc: float, tenure: float, products: int) -> tuple:
    """Return (rsi_value_0_100, rsi_category) for a single customer."""
    engagement = 0.7 * active + 0.3 * cc
    prod_depth = _PROD_MAP.get(products, 0.25)
    loyalty    = min(tenure / 10, 1)
    rsi = round((0.4 * engagement + 0.3 * prod_depth + 0.3 * loyalty) * 100, 1)

    if rsi <= 30:
        cat = 'High Risk'
    elif rsi <= 60:
        cat = 'Moderate Risk'
    elif rsi <= 80:
        cat = 'Stable'
    else:
        cat = 'Loyal'
    return rsi, cat


def _risk_level(prob: float) -> str:
    if prob > CHURN_HIGH_THRESHOLD:
        return 'High Risk'
    if prob > CHURN_MODERATE_THRESHOLD:
        return 'Moderate Risk'
    return 'Low Risk'


# ══════════════════════════════════════════════════════════════════════════
# MODEL BUNDLE LOADER
# ══════════════════════════════════════════════════════════════════════════

def load_model(filename: str = None):
    """
    Load the saved model bundle. Returns (model, scaler, feature_cols, decision_threshold).

    Older bundles that don't carry a decision_threshold fall back to the
    config default so we never crash on a legacy model file.
    """
    filename = filename or DEFAULT_MODEL_NAME
    path = Path(MODELS_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            f"Run the pipeline (python main.py) first to train and save the model."
        )
    bundle = joblib.load(path)
    threshold = bundle.get('decision_threshold')
    if threshold is None:
        from src.config import DECISION_THRESHOLD as default_threshold
        threshold = default_threshold
    return bundle['model'], bundle['scaler'], bundle['features'], float(threshold)


# ══════════════════════════════════════════════════════════════════════════
# FEATURE BUILDER (was predict.build_features — verbatim)
# ══════════════════════════════════════════════════════════════════════════

def build_features(customer: dict) -> pd.DataFrame:
    """
    Given a raw customer dict (with original fields), compute all
    engineered features and return a single-row DataFrame ready for prediction.
    """
    row = pd.Series(customer)

    balance      = float(row.get('Balance', 0))
    salary       = float(row.get('EstimatedSalary', 1))
    tenure       = float(row.get('Tenure', 0))
    products     = float(row.get('NumOfProducts', 1))
    active       = float(row.get('IsActiveMember', 0))
    cc           = float(row.get('HasCrCard', 0))
    credit_score = float(row.get('CreditScore', 650))

    features = {
        'CreditScore':          credit_score,
        'Age':                  float(row.get('Age', 40)),
        'Tenure':               tenure,
        'Balance':              balance,
        'NumOfProducts':        products,
        'HasCrCard':            cc,
        'IsActiveMember':       active,
        'EstimatedSalary':      salary,
        'BalanceSalaryRatio':   balance / (salary + 1),
        'ProductsPerTenure':    products / (tenure + 1),
        'EngagementScore':      0.7 * active + 0.3 * cc,
        'WealthScore':          (balance / 250898 + salary / 199992 + credit_score / 850),
        'RelationshipStrength': (
            0.4 * (0.7 * active + 0.3 * cc) +
            0.3 * ((products - 1) / 3) +
            0.3 * min(tenure / 10, 1)
        ),
        'Geography_enc':        GEO_MAP.get(row.get('Geography', 'France'), 0),
        'Gender_enc':           GEN_MAP.get(row.get('Gender', 'Male'), 1),
    }
    return pd.DataFrame([features])


# ══════════════════════════════════════════════════════════════════════════
# PREDICT_CHURN  (was src/predict.predict_churn — verbatim body)
# ══════════════════════════════════════════════════════════════════════════

def predict_churn(customer: dict):
    """
    Predict churn probability and return full result dict.

    Args:
        customer: dict with keys matching the European Bank dataset columns.

    Returns:
        dict with keys: churn_probability (%), risk_level, rsi, rsi_category, recommendation.
    """
    model, scaler, feature_cols, decision_threshold = load_model()

    X = build_features(customer)[feature_cols]
    X_sc = scaler.transform(X)
    prob = float(model.predict_proba(X_sc)[0, 1])

    # RSI
    active   = float(customer.get('IsActiveMember', 0))
    cc       = float(customer.get('HasCrCard', 0))
    tenure   = float(customer.get('Tenure', 0))
    products = int(customer.get('NumOfProducts', 1))
    rsi, rsi_cat = _rsi_for(active, cc, tenure, products)

    # Risk level
    risk_level = _risk_level(prob)

    # Recommendation (uses the single rule table, with the customer row)
    row = pd.Series(customer)
    row['ChurnProbability'] = prob
    rec = _recommend(row)

    return {
        'churn_probability':   round(prob * 100, 1),
        'predicted_churn':     bool(prob >= decision_threshold),
        'decision_threshold':  decision_threshold,
        'risk_level':          risk_level,
        'rsi':                 rsi,
        'rsi_category':        rsi_cat,
        'recommendation':      rec,
        'revenue_at_risk':     round(float(customer.get('Balance', 0)) * prob, 2),
    }


# ══════════════════════════════════════════════════════════════════════════
# PREDICT_SINGLE  (was src/utils.predict_single — verbatim body, value-equal)
# ══════════════════════════════════════════════════════════════════════════

def predict_single(customer: dict, model, scaler, feature_cols: list,
                   decision_threshold: float = None) -> dict:
    """
    Predict churn for a single customer dict using the already-loaded model.
    Uses build_features (the single-customer path) so the engineered
    features are value-identical to what `predict_churn` produces — using
    engineer_features on a 1-row frame collapses MinMax scalers to 0.
    Returns: churn_probability, predicted_churn, decision_threshold,
    risk_level, rsi, rsi_category, recommendation, revenue_at_risk.
    """
    if decision_threshold is None:
        from src.config import DECISION_THRESHOLD as _default
        decision_threshold = _default

    X = build_features(customer)[feature_cols]
    X_sc = scaler.transform(X)
    prob = float(model.predict_proba(X_sc)[0, 1])

    # RSI
    active   = float(customer.get('IsActiveMember', 0))
    cc       = float(customer.get('HasCrCard', 0))
    tenure   = float(customer.get('Tenure', 0))
    products = int(customer.get('NumOfProducts', 1))
    rsi, rsi_cat = _rsi_for(active, cc, tenure, products)

    risk_level = _risk_level(prob)

    row_with_prob = pd.Series(customer)
    row_with_prob['ChurnProbability'] = prob
    rec = _recommend(row_with_prob)

    return {
        'churn_probability':   round(prob * 100, 1),
        'predicted_churn':     bool(prob >= decision_threshold),
        'decision_threshold':  decision_threshold,
        'risk_level':          risk_level,
        'rsi':                 rsi,
        'rsi_category':        rsi_cat,
        'recommendation':      rec,
        'revenue_at_risk':     round(float(customer.get('Balance', 0)) * prob, 2),
    }


# ══════════════════════════════════════════════════════════════════════════
# CLI demo (was src/predict.__main__ — verbatim)
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    example_customer = {
        'CreditScore':     580,
        'Geography':       'Germany',
        'Gender':          'Female',
        'Age':             52,
        'Tenure':          2,
        'Balance':         145000,
        'NumOfProducts':   1,
        'HasCrCard':       1,
        'IsActiveMember':  0,
        'EstimatedSalary': 98000,
    }
    result = predict_churn(example_customer)
    print("\nExample prediction:")
    print(f"  Churn probability : {result['churn_probability']}%")
    print(f"  Risk level        : {result['risk_level']}")
    print(f"  RSI               : {result['rsi']} ({result['rsi_category']})")
    print(f"  Recommendation    : {result['recommendation']}")
