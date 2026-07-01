"""
preprocessing.py
Data ingestion, validation, and cleaning for the European Bank dataset.
"""

import pandas as pd
import numpy as np


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load the European Bank CSV, validate key columns,
    handle missing values, and return a clean DataFrame.
    """
    df = pd.read_csv(filepath)

    # ── Standardise column names ───────────────────────────────────────────
    df.columns = df.columns.str.strip()

    # ── Drop columns not needed for analysis ──────────────────────────────
    drop_cols = [c for c in ['RowNumber', 'CustomerId', 'Surname', 'Year'] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # ── Validate required columns ──────────────────────────────────────────
    required = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
        'EstimatedSalary', 'Exited'
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # ── Handle missing values ──────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # ── Type enforcement ───────────────────────────────────────────────────
    binary_cols = ['HasCrCard', 'IsActiveMember', 'Exited']
    for col in binary_cols:
        df[col] = df[col].astype(int)

    df['NumOfProducts'] = df['NumOfProducts'].astype(int)
    df['CreditScore'] = df['CreditScore'].astype(int)
    df['Age'] = df['Age'].astype(int)
    df['Tenure'] = df['Tenure'].astype(int)
    df['Balance'] = df['Balance'].astype(float)
    df['EstimatedSalary'] = df['EstimatedSalary'].astype(float)

    # ── Outlier capping (IQR method for numeric features) ─────────────────
    for col in ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']:
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=q1, upper=q99)

    # ── Validate binary variables ──────────────────────────────────────────
    for col in binary_cols:
        assert df[col].isin([0, 1]).all(), f"Non-binary values found in {col}"

    assert df['Exited'].mean() > 0.05, "Churn rate suspiciously low — check data"
    assert len(df) >= 100, "Dataset too small — check file path"

    print(f"  [preprocessing] Loaded {len(df):,} rows, {df.shape[1]} columns.")
    print(f"  [preprocessing] Churn rate: {df['Exited'].mean()*100:.1f}%")
    print(f"  [preprocessing] Missing values after cleaning: {df.isna().sum().sum()}")

    return df
