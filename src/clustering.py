"""
clustering.py
KMeans customer segmentation (k=4) and RSI computation.
Called from notebooks and main.py.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


CLUSTER_NAMES = ['Young Active', 'Premium Loyal', 'Wealthy Disengaged', 'High Risk']

CLUSTER_COLORS = {
    'Young Active':       '#3266ad',
    'Premium Loyal':      '#1d9e75',
    'Wealthy Disengaged': '#f0a500',
    'High Risk':          '#c0392b',
}

SEG_FEATURES = [
    'Age', 'Balance', 'IsActiveMember', 'NumOfProducts',
    'Tenure', 'EngagementScore', 'WealthScore', 'RelationshipStrength'
]


def segment_customers(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42):
    """
    Fit KMeans on behavioural features, assign named cluster labels,
    return enriched DataFrame + summary table.
    """
    df = df.copy()
    X = df[SEG_FEATURES].copy()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    raw_labels = km.fit_predict(X_sc)

    # Rank raw clusters by (EngagementScore centre + Balance centre)
    # to get consistent naming regardless of random init
    centres = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=SEG_FEATURES
    )
    rank = centres['EngagementScore'] + centres['Balance'] / 1e6
    sorted_ids = rank.argsort().values       # ascending → index 0 = lowest
    name_order = ['High Risk', 'Wealthy Disengaged', 'Young Active', 'Premium Loyal']
    id_to_name = {sorted_ids[i]: name_order[i] for i in range(n_clusters)}

    df['Cluster'] = [id_to_name[l] for l in raw_labels]

    summary = df.groupby('Cluster').agg(
        Count       =('Exited', 'count'),
        ChurnRate   =('Exited', lambda x: round(x.mean() * 100, 1)),
        AvgBalance  =('Balance', lambda x: round(x.mean(), 0)),
        AvgAge      =('Age', lambda x: round(x.mean(), 1)),
        AvgProducts =('NumOfProducts', lambda x: round(x.mean(), 2)),
        ActiveRate  =('IsActiveMember', lambda x: round(x.mean() * 100, 1)),
    ).reset_index()

    return df, summary, X_sc, km


def compute_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Retention Strength Index (0–100) for every customer.

    RSI = (0.4 × Engagement + 0.3 × ProductDepth + 0.3 × Loyalty) × 100
    """
    df = df.copy()
    prod_map = {1: 0.25, 2: 1.0, 3: 0.20, 4: 0.0}
    product_depth = df['NumOfProducts'].map(prod_map).fillna(0.25)
    loyalty = (df['Tenure'] / max(df['Tenure'].max(), 1)).clip(0, 1)

    df['RSI'] = (
        0.4 * df['EngagementScore'] +
        0.3 * product_depth +
        0.3 * loyalty
    ) * 100
    df['RSI'] = df['RSI'].round(1)

    df['RSICategory'] = pd.cut(
        df['RSI'],
        bins=[-np.inf, 30, 60, 80, np.inf],
        labels=['High Risk', 'Moderate Risk', 'Stable', 'Loyal']
    ).astype(str)

    return df


def get_recommendation(row) -> str:
    """Rule-based retention recommendation for a single customer row."""
    prob    = row.get('ChurnProbability', 0)
    active  = row['IsActiveMember']
    prods   = row['NumOfProducts']
    balance = row['Balance']

    if prob > 0.80:
        return 'Immediate Outreach'
    elif active == 0:
        return 'Reactivation Campaign'
    elif prods == 1:
        return 'Cross-Sell Programme'
    elif balance > 97199:
        return 'Relationship Manager Assignment'
    else:
        return 'Standard Retention Programme'
