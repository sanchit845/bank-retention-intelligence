"""
retention_index.py
Computes the Retention Strength Index (RSI) for every customer.

Formula:
  RSI = (0.4 × EngagementScore + 0.3 × ProductDepth + 0.3 × Loyalty) × 100

Scale:
   0–30  → High Risk
  31–60  → Moderate Risk
  61–80  → Stable
  81–100 → Loyal
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def compute_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI and RSICategory columns to the DataFrame.
    All inputs are normalised to [0, 1] before weighting.
    """
    df = df.copy()

    # ── Engagement component (0–1, already in that range) ────────────────
    engagement = df['EngagementScore'].clip(0, 1)

    # ── Product depth: 2 products is optimal; penalise 1, 3, 4 ──────────
    # Map: 1→0.25, 2→1.0, 3→0.20, 4→0.0
    prod_map = {1: 0.25, 2: 1.0, 3: 0.20, 4: 0.0}
    product_depth = df['NumOfProducts'].map(prod_map).fillna(0.0)

    # ── Loyalty = normalised tenure ───────────────────────────────────────
    max_tenure = df['Tenure'].max() if df['Tenure'].max() > 0 else 10
    loyalty = (df['Tenure'] / max_tenure).clip(0, 1)

    # ── RSI formula ───────────────────────────────────────────────────────
    df['RSI'] = (
        0.4 * engagement +
        0.3 * product_depth +
        0.3 * loyalty
    ) * 100

    df['RSI'] = df['RSI'].round(1)

    # ── Category ─────────────────────────────────────────────────────────
    def categorise(rsi):
        if rsi <= 30:
            return 'High Risk'
        elif rsi <= 60:
            return 'Moderate Risk'
        elif rsi <= 80:
            return 'Stable'
        else:
            return 'Loyal'

    df['RSICategory'] = df['RSI'].apply(categorise)

    print(f"  [retention_index] RSI distribution:")
    dist = df['RSICategory'].value_counts()
    for cat, count in dist.items():
        avg_rsi = df[df['RSICategory'] == cat]['RSI'].mean()
        churn_rate = df[df['RSICategory'] == cat]['Exited'].mean() * 100
        print(f"    {cat:16s}: {count:5,}  avg RSI={avg_rsi:.1f}  churn={churn_rate:.1f}%")

    return df
