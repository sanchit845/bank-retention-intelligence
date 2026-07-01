"""
visualization.py
Reusable chart functions used across all notebooks.
Returns matplotlib Figure objects — call plt.show() or fig.savefig() in the notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Global style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f9f9f9',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.family':      'DejaVu Sans',
})

PALETTE = {
    'blue':   '#3266ad',
    'red':    '#c0392b',
    'amber':  '#f0a500',
    'green':  '#1d9e75',
    'gray':   '#73726c',
    'purple': '#7f77dd',
    'dark':   '#8B0000',
}

CLUSTER_COLORS = {
    'Young Active':       PALETTE['blue'],
    'Premium Loyal':      PALETTE['green'],
    'Wealthy Disengaged': PALETTE['amber'],
    'High Risk':          PALETTE['red'],
}


# ── EDA Charts ────────────────────────────────────────────────────────────

def plot_churn_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = df['Exited'].value_counts()
    axes[0].bar(['Retained', 'Churned'], counts.values,
                color=[PALETTE['blue'], PALETTE['red']], width=0.5, edgecolor='white')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 80, f'{v:,}', ha='center', fontsize=11, fontweight='bold')
    axes[0].set_title('Churn Count', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Customers')
    axes[0].set_ylim(0, max(counts.values) * 1.12)

    axes[1].pie(counts.values, labels=['Retained', 'Churned'],
                colors=[PALETTE['blue'], PALETTE['red']],
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Churn Rate', fontsize=13, fontweight='bold')
    fig.suptitle('Churn Distribution', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_feature_distributions(df):
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(df[df['Exited'] == 0][col], bins=30, alpha=0.6,
                     color=PALETTE['blue'], label='Retained', edgecolor='white')
        axes[i].hist(df[df['Exited'] == 1][col], bins=30, alpha=0.6,
                     color=PALETTE['red'], label='Churned', edgecolor='white')
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].legend(fontsize=9)
    axes[-1].set_visible(False)
    fig.suptitle('Feature Distributions by Churn Status', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_categorical_churn(df):
    cats = ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(cats):
        grp = df.groupby(col)['Exited'].mean() * 100
        bars = axes[i].bar(grp.index.astype(str), grp.values,
                           color=PALETTE['blue'], edgecolor='white', width=0.55)
        for bar, val in zip(bars, grp.values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
        axes[i].set_title(f'Churn by {col}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Churn Rate (%)')
        axes[i].set_xticklabels(grp.index.astype(str), rotation=20)
    axes[-1].set_visible(False)
    fig.suptitle('Churn Rate by Categorical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    numeric = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(11, 8))
    mask = np.triu(np.ones_like(numeric.corr(), dtype=bool))
    sns.heatmap(numeric.corr(), mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, ax=ax, square=True,
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ── Engagement Charts ─────────────────────────────────────────────────────

def plot_engagement_vs_churn(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Active vs inactive
    act = df.groupby('IsActiveMember')['Exited'].mean() * 100
    axes[0].bar(['Inactive', 'Active'], act.values,
                color=[PALETTE['red'], PALETTE['blue']], width=0.5, edgecolor='white')
    for i, v in enumerate(act.values):
        axes[0].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Churn Rate (%)')
    axes[0].set_ylim(0, 33)
    axes[0].set_title('Active vs Inactive Churn', fontsize=12, fontweight='bold')

    # Credit card stickiness
    cc = df.groupby('HasCrCard')['Exited'].mean() * 100
    axes[1].bar(['No Card', 'Has Card'], cc.values,
                color=[PALETTE['gray'], PALETTE['blue']], width=0.5, edgecolor='white')
    for i, v in enumerate(cc.values):
        axes[1].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Churn Rate (%)')
    axes[1].set_ylim(18, 24)
    axes[1].set_title('Credit Card Stickiness', fontsize=12, fontweight='bold')

    fig.suptitle('Engagement Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ── Product Charts ────────────────────────────────────────────────────────

def plot_product_vs_churn(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    prod = df.groupby('NumOfProducts')['Exited'].mean() * 100
    colors = [PALETTE['amber'], PALETTE['blue'], PALETTE['red'], PALETTE['dark']]
    bars = axes[0].bar(prod.index, prod.values,
                       color=colors[:len(prod)], edgecolor='white', width=0.6)
    for bar, val in zip(bars, prod.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Number of Products'); axes[0].set_ylabel('Churn Rate (%)')
    axes[0].set_title('Product Count vs Churn Rate', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 115)

    counts = df['NumOfProducts'].value_counts().sort_index()
    axes[1].pie(counts.values,
                labels=[f'{i} product(s)' for i in counts.index],
                colors=[PALETTE['blue'], PALETTE['green'], PALETTE['amber'], PALETTE['red']],
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Product Adoption Distribution', fontsize=12, fontweight='bold')

    fig.suptitle('Product Utilisation Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ── Segmentation Charts ───────────────────────────────────────────────────

def plot_clusters_pca(X_sc, labels):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_sc)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, color in CLUSTER_COLORS.items():
        mask = np.array(labels) == name
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, alpha=0.55, s=18, label=name, linewidths=0)
    ax.set_title('Customer Segments — PCA Projection', fontsize=13, fontweight='bold')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(title='Segment', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig


def plot_segment_profiles(summary):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    clusters = summary['Cluster'].tolist()
    colors = [CLUSTER_COLORS.get(c, PALETTE['blue']) for c in clusters]

    axes[0].barh(clusters, summary['ChurnRate'], color=colors, edgecolor='white', height=0.55)
    axes[0].set_xlabel('Churn Rate (%)'); axes[0].set_title('Churn Rate', fontsize=12, fontweight='bold')
    for i, v in enumerate(summary['ChurnRate']):
        axes[0].text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=10)

    axes[1].barh(clusters, summary['AvgBalance'], color=colors, edgecolor='white', height=0.55)
    axes[1].set_xlabel('Avg Balance (€)'); axes[1].set_title('Average Balance', fontsize=12, fontweight='bold')

    axes[2].barh(clusters, summary['ActiveRate'], color=colors, edgecolor='white', height=0.55)
    axes[2].set_xlabel('Active Rate (%)'); axes[2].set_title('Active Member Rate', fontsize=12, fontweight='bold')

    fig.suptitle('Customer Segment Profiles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ── Model Evaluation Charts ───────────────────────────────────────────────

def plot_model_comparison(results_df):
    fig, ax = plt.subplots(figsize=(11, 5))
    metrics = ['Accuracy', 'Precision', 'Recall', 'ROC_AUC']
    colors  = [PALETTE['blue'], PALETTE['green'], PALETTE['amber'], PALETTE['red']]
    x = np.arange(len(results_df))
    w = 0.2
    for i, (m, c) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * w, results_df[m], width=w, label=m, color=c, alpha=0.88)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(results_df['Model'], rotation=20, ha='right')
    ax.set_ylabel('Score (%)'); ax.set_ylim(0, 105)
    ax.set_title('Model Comparison — All Metrics', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig


def plot_roc_curve(fpr, tpr, auc_val):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=PALETTE['blue'], lw=2, label=f'CatBoost (AUC={auc_val:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random baseline')
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE['blue'])
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Retained', 'Churned'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Retained', 'Churned'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    plt.tight_layout()
    return fig


def plot_feature_importance(feature_names, importances):
    imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    imp = imp.sort_values('Importance', ascending=True)
    colors = [PALETTE['red'] if i >= len(imp) - 5 else PALETTE['blue']
              for i in range(len(imp))]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp['Feature'], imp['Importance'], color=colors, height=0.65)
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance', fontsize=13, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig


# ── RSI & Retention Charts ────────────────────────────────────────────────

def plot_rsi_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df['RSI'], bins=30, color=PALETTE['blue'], edgecolor='white', alpha=0.85)
    for thresh, color, label in [(30, PALETTE['red'], 'High Risk'),
                                  (60, PALETTE['amber'], 'Moderate'),
                                  (80, PALETTE['green'], 'Stable')]:
        axes[0].axvline(thresh, color=color, linestyle='--', lw=1.5, label=label)
    axes[0].set_xlabel('RSI Score'); axes[0].set_ylabel('Customers')
    axes[0].set_title('RSI Distribution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)

    cat_order = ['High Risk', 'Moderate Risk', 'Stable', 'Loyal']
    rsi_churn = df.groupby('RSICategory')['Exited'].mean() * 100
    rsi_churn = rsi_churn.reindex([c for c in cat_order if c in rsi_churn.index])
    colors = [PALETTE['red'], PALETTE['amber'], PALETTE['blue'], PALETTE['green']]
    bars = axes[1].bar(rsi_churn.index, rsi_churn.values,
                       color=colors[:len(rsi_churn)], width=0.5, edgecolor='white')
    for bar, val in zip(bars, rsi_churn.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                     f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Churn Rate (%)'); axes[1].set_xticklabels(rsi_churn.index, rotation=15)
    axes[1].set_title('Churn Rate by RSI Category', fontsize=12, fontweight='bold')

    fig.suptitle('Retention Strength Index Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
