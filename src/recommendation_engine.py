"""
recommendation_engine.py
Rule-based retention recommendation engine.
Generates a personalised action recommendation for each customer.
"""

import pandas as pd
import numpy as np

GEO_MAP = {'France': 0, 'Spain': 1, 'Germany': 2}
GEN_MAP = {'Female': 0, 'Male': 1}


def _get_churn_proba(model, scaler, df, feature_cols):
    """Return predicted churn probabilities for the full dataset."""
    dfc = df.copy()
    dfc['Geography_enc'] = dfc['Geography'].map(GEO_MAP).fillna(0).astype(int)
    dfc['Gender_enc'] = dfc['Gender'].map(GEN_MAP).fillna(0).astype(int)
    X = dfc[feature_cols].fillna(0)
    X_sc = scaler.transform(X)
    return model.predict_proba(X_sc)[:, 1]


def recommend(row):
    """
    Priority-ordered rules — first match wins.

    Rules (from project specification):
      1. churn_proba > 0.80              → Immediate Outreach
      2. IsActiveMember == 0             → Reactivation Campaign
      3. NumOfProducts == 1              → Cross-Sell Programme
      4. Balance > 97199 (high value)    → Relationship Manager Assignment
      5. Default                         → Standard Retention Programme
    """
    prob = row.get('ChurnProbability', 0)
    active = row['IsActiveMember']
    products = row['NumOfProducts']
    balance = row['Balance']
    HIGH_VALUE_THRESHOLD = 97199  # median balance

    if prob > 0.80:
        return 'Immediate Outreach'
    elif active == 0:
        return 'Reactivation Campaign'
    elif products == 1:
        return 'Cross-Sell Programme'
    elif balance > HIGH_VALUE_THRESHOLD:
        return 'Relationship Manager Assignment'
    else:
        return 'Standard Retention Programme'


def generate_recommendations(df, model, scaler, feature_cols):
    """
    Add ChurnProbability and Recommendation columns to the DataFrame.
    """
    df = df.copy()

    # Get model predictions
    probas = _get_churn_proba(model, scaler, df, feature_cols)
    df['ChurnProbability'] = probas.round(4)
    df['PredictedChurn'] = (probas > 0.5).astype(int)

    # Apply recommendation rules
    df['Recommendation'] = df.apply(recommend, axis=1)

    # Priority segment flags
    df['IsImmediateOutreach'] = (df['ChurnProbability'] > 0.80).astype(int)

    print(f"  [recommendation] Distribution:")
    rec_dist = df['Recommendation'].value_counts()
    for rec, count in rec_dist.items():
        print(f"    {rec:40s}: {count:,}")

    high_risk_count = df[df['ChurnProbability'] > 0.80].shape[0]
    print(f"  [recommendation] Immediate outreach candidates: {high_risk_count:,}")

    return df


def get_individual_recommendation(customer_dict, model, scaler, feature_cols):
    """
    Generate a recommendation for a single customer (dict or Series).
    Returns a dict with probability, risk level, RSI, and recommendation.
    """
    row = pd.Series(customer_dict)
    df_single = pd.DataFrame([row])

    dfc = df_single.copy()
    dfc['Geography_enc'] = dfc['Geography'].map(GEO_MAP).fillna(0).astype(int)
    dfc['Gender_enc'] = dfc['Gender'].map(GEN_MAP).fillna(0).astype(int)

    cols_needed = [c for c in feature_cols if c in dfc.columns]
    for c in feature_cols:
        if c not in dfc.columns:
            dfc[c] = 0

    X = dfc[feature_cols].fillna(0)
    X_sc = scaler.transform(X)
    prob = float(model.predict_proba(X_sc)[0, 1])

    row_with_prob = row.copy()
    row_with_prob['ChurnProbability'] = prob
    rec = recommend(row_with_prob)

    if prob > 0.80:
        risk_level = 'High Risk'
    elif prob > 0.50:
        risk_level = 'Moderate Risk'
    else:
        risk_level = 'Low Risk'

    return {
        'churn_probability': round(prob * 100, 1),
        'risk_level': risk_level,
        'recommendation': rec
    }
