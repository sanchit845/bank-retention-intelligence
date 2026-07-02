"""
explainability.py
SHAP-based explainability for the best trained model.

Moved verbatim from the previous src/shap_analysis.py — only the module
location and one path constant have changed.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, DATA_PROCESSED_DIR, SAMPLE_SHAP_SIZE, RANDOM_STATE
from src.modeling import GEO_MAP, GEN_MAP


def run_shap(model, scaler, df, feature_cols, output_dir: str = None):
    """
    Global and individual SHAP explainability for the best model.
    Writes data/processed/figures/shap_summary.png and
    data/processed/shap_feature_importance.csv.

    The `output_dir` argument is preserved for back-compat but the SHAP
    figures always land in FIGURES_DIR — keeping all generated charts
    in one canonical place.
    """
    # output_dir is intentionally unused; kept for API stability.
    _ = output_dir
    figures_dir = FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        import shap
    except ImportError:
        print("  [shap] shap not installed — skipping SHAP analysis.")
        return

    dfc = df.copy()
    dfc['Geography_enc'] = dfc['Geography'].map(GEO_MAP).fillna(0).astype(int)
    dfc['Gender_enc']    = dfc['Gender'].map(GEN_MAP).fillna(0).astype(int)

    X = dfc[feature_cols].fillna(0)
    X_scaled = scaler.transform(X)

    # Use a sample for speed if dataset is large
    sample_size = min(SAMPLE_SHAP_SIZE, len(X_scaled))
    np.random.seed(RANDOM_STATE)
    idx = np.random.choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled[idx]

    print(f"  [shap] Computing SHAP values on {sample_size} samples...")

    # ── TreeExplainer (fast for tree-based models) ────────────────────────
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        # Handle 2-class output (some models return list)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        # Fallback to KernelExplainer
        background = shap.sample(X_scaled, 100, random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    # ── Global SHAP summary plot ──────────────────────────────────────────
    X_sample_df = pd.DataFrame(X_sample, columns=feature_cols)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X_sample_df, feature_names=feature_cols,
        plot_type='bar', show=False, color='#3266ad'
    )
    plt.title('SHAP Feature Importance — Global Churn Drivers', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / 'shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Mean absolute SHAP values to CSV ─────────────────────────────────
    mean_shap = pd.DataFrame({
        'Feature': feature_cols,
        'MeanAbsSHAP': np.abs(shap_values).mean(axis=0)
    }).sort_values('MeanAbsSHAP', ascending=False)

    mean_shap.to_csv(DATA_PROCESSED_DIR / 'shap_feature_importance.csv', index=False)

    print(f"  [shap] Top 5 churn drivers:")
    for _, row in mean_shap.head(5).iterrows():
        print(f"    {row['Feature']:30s}: {row['MeanAbsSHAP']:.4f}")

    return shap_values, explainer
