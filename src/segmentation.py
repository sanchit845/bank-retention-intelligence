"""
segmentation.py
KMeans customer segmentation into 4 behavioural clusters.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


CLUSTER_NAMES = {
    0: 'Young Active',
    1: 'Premium Loyal',
    2: 'Wealthy Disengaged',
    3: 'High Risk'
}

CLUSTER_COLORS = {
    'Young Active': '#3266ad',
    'Premium Loyal': '#1d9e75',
    'Wealthy Disengaged': '#f0a500',
    'High Risk': '#c0392b'
}


def segment_customers(df: pd.DataFrame, output_dir: str):
    """
    Fit KMeans (k=4) on behavioural features, assign cluster labels,
    generate PCA visualisation, and return the enriched DataFrame.
    """
    seg_features = [
        'Age', 'Balance', 'IsActiveMember', 'NumOfProducts',
        'Tenure', 'EngagementScore', 'WealthScore', 'RelationshipStrength'
    ]

    X_seg = df[seg_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seg)

    # ── Fit KMeans ────────────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    raw_labels = kmeans.fit_predict(X_scaled)

    # ── Map cluster IDs to meaningful names ───────────────────────────────
    # Sort clusters by RelationshipStrength centre to get consistent ordering
    centres = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=seg_features
    )
    # Rank clusters for stable naming: by EngagementScore then Balance
    rank_series = centres['EngagementScore'] + centres['Balance'] / 1e6
    sorted_ids = rank_series.argsort().values   # ascending

    # Build a remapping: sorted_ids[0] → 0 (High Risk), ..., [3] → 3 (Premium)
    name_order = ['High Risk', 'Wealthy Disengaged', 'Young Active', 'Premium Loyal']
    id_to_name = {sorted_ids[i]: name_order[i] for i in range(4)}

    df = df.copy()
    df['Cluster'] = [id_to_name[l] for l in raw_labels]

    # ── Segment summary ───────────────────────────────────────────────────
    seg_summary = df.groupby('Cluster').agg(
        Count=('Exited', 'count'),
        ChurnRate=('Exited', 'mean'),
        AvgBalance=('Balance', 'mean'),
        AvgAge=('Age', 'mean'),
        AvgProducts=('NumOfProducts', 'mean'),
        ActiveRate=('IsActiveMember', 'mean'),
        AvgEngagement=('EngagementScore', 'mean')
    ).reset_index()
    seg_summary['ChurnRate'] = (seg_summary['ChurnRate'] * 100).round(1)
    seg_summary['AvgBalance'] = seg_summary['AvgBalance'].round(0)
    seg_summary['AvgAge'] = seg_summary['AvgAge'].round(1)
    seg_summary['AvgProducts'] = seg_summary['AvgProducts'].round(2)
    seg_summary['ActiveRate'] = (seg_summary['ActiveRate'] * 100).round(1)

    seg_summary.to_csv(os.path.join(output_dir, 'segment_summary.csv'), index=False)
    print(f"  [segmentation] Segment summary:\n{seg_summary[['Cluster','Count','ChurnRate','AvgBalance']].to_string(index=False)}")

    # ── PCA plot ──────────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8f8f8')

    for cname, color in CLUSTER_COLORS.items():
        mask = df['Cluster'] == cname
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, alpha=0.55, s=18, label=cname, linewidths=0
        )

    ax.set_title('Customer Segments — PCA Projection', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Principal Component 1', fontsize=10)
    ax.set_ylabel('Principal Component 2', fontsize=10)
    ax.legend(title='Segment', fontsize=9, title_fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'customer_segments.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return df, seg_summary
