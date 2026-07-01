"""
evaluate_model.py
Full model evaluation: metrics, confusion matrix, ROC curve, PR curve,
feature importance chart.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

GEO_MAP = {'France': 0, 'Spain': 1, 'Germany': 2}
GEN_MAP = {'Female': 0, 'Male': 1}


def evaluate_model(model, scaler, df, feature_cols, output_dir):
    dfc = df.copy()
    dfc['Geography_enc'] = dfc['Geography'].map(GEO_MAP).fillna(0).astype(int)
    dfc['Gender_enc'] = dfc['Gender'].map(GEN_MAP).fillna(0).astype(int)

    X = dfc[feature_cols].fillna(0)
    y = dfc['Exited']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_sc = scaler.transform(X_test)

    y_pred = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    metrics = {
        'accuracy':   accuracy_score(y_test, y_pred),
        'precision':  precision_score(y_test, y_pred),
        'recall':     recall_score(y_test, y_pred),
        'f1':         f1_score(y_test, y_pred),
        'roc_auc':    roc_auc_score(y_test, y_proba),
        'pr_auc':     average_precision_score(y_test, y_proba),
    }

    print(f"  [evaluate] Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  [evaluate] Precision : {metrics['precision']*100:.2f}%")
    print(f"  [evaluate] Recall    : {metrics['recall']*100:.2f}%")
    print(f"  [evaluate] F1 Score  : {metrics['f1']:.4f}")
    print(f"  [evaluate] ROC-AUC   : {metrics['roc_auc']*100:.2f}%")
    print(f"  [evaluate] PR-AUC    : {metrics['pr_auc']*100:.2f}%")

    # ── Confusion Matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    classes = ['Retained', 'Churned']
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── ROC Curve ────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#3266ad', lw=2, label=f'ROC-AUC = {metrics["roc_auc"]:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random baseline')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#3266ad')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve — CatBoost', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Feature Importance ───────────────────────────────────────────────
    if hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        imp.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

        colors = ['#c0392b' if i >= len(imp) - 5 else '#3266ad' for i in range(len(imp))]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp['Feature'], imp['Importance'], color=colors, height=0.65)
        ax.set_xlabel('Feature Importance', fontsize=11)
        ax.set_title('Feature Importance — Top Churn Drivers', fontsize=13, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'figures', 'feature_importance.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # ── Engagement vs Churn chart ─────────────────────────────────────────
    engagement_churn = df.groupby('IsActiveMember')['Exited'].mean() * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(['Inactive', 'Active'], engagement_churn.values,
                  color=['#c0392b', '#3266ad'], width=0.5)
    for bar, val in zip(bars, engagement_churn.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Churn Rate (%)', fontsize=11)
    ax.set_title('Engagement vs Churn Rate', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 35)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'engagement_vs_churn.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Products vs Churn chart ───────────────────────────────────────────
    prod_churn = df.groupby('NumOfProducts')['Exited'].mean() * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    bar_colors = ['#f0a500', '#3266ad', '#c0392b', '#8B0000']
    bars = ax.bar(prod_churn.index, prod_churn.values,
                  color=bar_colors[:len(prod_churn)], width=0.6)
    for bar, val in zip(bars, prod_churn.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('Number of Products', fontsize=11)
    ax.set_ylabel('Churn Rate (%)', fontsize=11)
    ax.set_title('Product Count vs Churn Rate', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'products_vs_churn.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return metrics
