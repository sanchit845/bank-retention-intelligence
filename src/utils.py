"""
utils.py
Shared utility functions used across notebooks and main.py.
"""

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime


# ── Paths ─────────────────────────────────────────────────────────────────

def get_project_root() -> str:
    """Return absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_path(*parts) -> str:
    """Build an absolute path relative to the project root."""
    return os.path.join(get_project_root(), *parts)


# ── Model I/O ─────────────────────────────────────────────────────────────

def save_model(model, scaler, feature_cols: list, filename: str = 'best_model.pkl'):
    """Save model bundle (model + scaler + features) to models/."""
    path = get_path('models', filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler, 'features': feature_cols}, path)
    print(f"  Model saved → {path}")
    return path


def load_model(filename: str = 'best_model.pkl'):
    """Load saved model bundle. Returns (model, scaler, feature_cols)."""
    path = get_path('models', filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}\nRun 02_model_development.ipynb first.")
    bundle = joblib.load(path)
    return bundle['model'], bundle['scaler'], bundle['features']


# ── Data I/O ──────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame, filename: str):
    """Save a DataFrame to data/processed/."""
    path = get_path('data', 'processed', filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved → {path}  ({len(df):,} rows)")


def load_processed(filename: str) -> pd.DataFrame:
    """Load a DataFrame from data/processed/."""
    path = get_path('data', 'processed', filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def save_figure(fig, filename: str, folder: str = 'reports/screenshots', dpi: int = 150):
    """Save a matplotlib figure to reports/screenshots/."""
    path = get_path(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"  Figure saved → {path}")


# ── Encoding helpers ──────────────────────────────────────────────────────

GEO_MAP = {'France': 0, 'Spain': 1, 'Germany': 2}
GEN_MAP = {'Female': 0, 'Male': 1}


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Add Geography_enc and Gender_enc columns."""
    df = df.copy()
    df['Geography_enc'] = df['Geography'].map(GEO_MAP).fillna(0).astype(int)
    df['Gender_enc']    = df['Gender'].map(GEN_MAP).fillna(0).astype(int)
    return df


# ── Prediction helpers ────────────────────────────────────────────────────

def predict_single(customer: dict, model, scaler, feature_cols: list) -> dict:
    """
    Predict churn for a single customer dict.
    Automatically engineers all features before predicting.
    Returns: churn_probability, risk_level, rsi, rsi_category, recommendation.
    """
    import pandas as pd
    from src.feature_engineering import engineer_features

    row = pd.DataFrame([customer])
    # Drop columns that might conflict
    for col in ['Geography_enc', 'Gender_enc']:
        if col in row.columns:
            row.drop(columns=[col], inplace=True)

    row = engineer_features(row)
    row = encode_categoricals(row)

    X = row[feature_cols].fillna(0)
    X_sc = scaler.transform(X)
    prob = float(model.predict_proba(X_sc)[0, 1])

    # RSI
    prod_map = {1: 0.25, 2: 1.0, 3: 0.20, 4: 0.0}
    active   = float(customer.get('IsActiveMember', 0))
    cc       = float(customer.get('HasCrCard', 0))
    tenure   = float(customer.get('Tenure', 0))
    products = int(customer.get('NumOfProducts', 1))
    rsi = round((
        0.4 * (0.7 * active + 0.3 * cc) +
        0.3 * prod_map.get(products, 0.25) +
        0.3 * min(tenure / 10, 1)
    ) * 100, 1)

    rsi_cat = ('High Risk' if rsi <= 30 else
               'Moderate Risk' if rsi <= 60 else
               'Stable' if rsi <= 80 else 'Loyal')
    risk_level = ('High Risk' if prob > 0.80 else
                  'Moderate Risk' if prob > 0.50 else 'Low Risk')

    balance = float(customer.get('Balance', 0))
    if prob > 0.80:          rec = 'Immediate Outreach'
    elif active == 0:        rec = 'Reactivation Campaign'
    elif products == 1:      rec = 'Cross-Sell Programme'
    elif balance > 97199:    rec = 'Relationship Manager Assignment'
    else:                    rec = 'Standard Retention Programme'

    return {
        'churn_probability': round(prob * 100, 1),
        'risk_level': risk_level,
        'rsi': rsi,
        'rsi_category': rsi_cat,
        'recommendation': rec,
    }


# ── Display helpers ───────────────────────────────────────────────────────

def print_section(title: str, width: int = 60):
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def print_metrics(metrics: dict):
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v*100:.2f}%")
        else:
            print(f"  {k:20s}: {v}")
