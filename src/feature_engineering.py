"""
feature_engineering.py
Creates all engineered features as defined in the project specification.
Called from notebooks and main.py.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 8 engineered features to the dataset:
      1. BalanceSalaryRatio
      2. ProductsPerTenure
      3. EngagementScore
      4. WealthScore
      5. RelationshipStrength
      6. AgeGroup (categorical)
      7. WealthSegment (categorical)
      8. HighValueDisengaged (binary flag)
    """
    df = df.copy()

    # 1. Balance / Salary Ratio
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)

    # 2. Products Per Tenure
    df['ProductsPerTenure'] = df['NumOfProducts'] / (df['Tenure'] + 1)

    # 3. Engagement Score
    df['EngagementScore'] = 0.7 * df['IsActiveMember'] + 0.3 * df['HasCrCard']

    # 4. Wealth Score (sum of min-max scaled components)
    scaler = MinMaxScaler()
    df['WealthScore'] = scaler.fit_transform(
        df[['Balance', 'EstimatedSalary', 'CreditScore']]
    ).sum(axis=1)

    # 5. Relationship Strength
    tenure_norm = MinMaxScaler().fit_transform(df[['Tenure']])[:, 0]
    products_norm = (df['NumOfProducts'] - 1) / 3
    df['RelationshipStrength'] = (
        0.4 * df['EngagementScore'] +
        0.3 * products_norm +
        0.3 * tenure_norm
    )

    # 6. Age Group
    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[0, 25, 35, 45, 55, 120],
        labels=['18-25', '26-35', '36-45', '46-55', '56+']
    ).astype(str)

    # 7. Wealth Segment
    if len(df) == 1:
        # Single customer — skip quantile binning, assign directly
        score = df['WealthScore'].iloc[0]
        max_score = 3.0  # max possible (3 features scaled 0-1, summed)
        if score < 0.75:
            df['WealthSegment'] = 'Low'
        elif score < 1.5:
            df['WealthSegment'] = 'Medium'
        elif score < 2.25:
            df['WealthSegment'] = 'High'
        else:
            df['WealthSegment'] = 'Premium'
    else:
        q = df['WealthScore'].quantile([0.25, 0.50, 0.75])
        df['WealthSegment'] = pd.cut(
            df['WealthScore'],
            bins=[-np.inf, q[0.25], q[0.50], q[0.75], np.inf],
            labels=['Low', 'Medium', 'High', 'Premium'],
            duplicates='drop'
        ).astype(str)

    # 8. High-Value Disengaged flag
    median_balance = df['Balance'].median()
    df['HighValueDisengaged'] = (
        (df['Balance'] > median_balance) & (df['IsActiveMember'] == 0)
    ).astype(int)

    return df


def get_feature_list() -> list:
    """Return the ML feature columns used for modelling."""
    return [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'BalanceSalaryRatio', 'ProductsPerTenure', 'EngagementScore',
        'WealthScore', 'RelationshipStrength', 'Geography_enc', 'Gender_enc'
    ]
