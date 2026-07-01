"""
data_cleaning.py
Handles loading, validating, and cleaning the European Bank dataset.
Called from notebooks and main.py.
"""

import pandas as pd
import numpy as np


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV, drop unused columns, fix types,
    fill missing values, cap outliers. Returns clean DataFrame.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Drop non-analytical columns
    drop_cols = [c for c in ['RowNumber', 'CustomerId', 'Surname', 'Year'] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # Required columns check
    required = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                'EstimatedSalary', 'Exited']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Fill missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Enforce types
    int_cols = ['CreditScore', 'Age', 'Tenure', 'NumOfProducts',
                'HasCrCard', 'IsActiveMember', 'Exited']
    for col in int_cols:
        df[col] = df[col].astype(int)
    df['Balance'] = df['Balance'].astype(float)
    df['EstimatedSalary'] = df['EstimatedSalary'].astype(float)

    # Outlier capping (1st–99th percentile)
    for col in ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']:
        df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))

    return df


def get_summary(df: pd.DataFrame) -> dict:
    """Return a quick summary dict for reporting."""
    return {
        'rows': len(df),
        'columns': df.shape[1],
        'churn_rate': round(df['Exited'].mean() * 100, 2),
        'missing_values': int(df.isna().sum().sum()),
        'active_rate': round(df['IsActiveMember'].mean() * 100, 2),
        'avg_balance': round(df['Balance'].mean(), 2),
        'avg_products': round(df['NumOfProducts'].mean(), 2),
    }
