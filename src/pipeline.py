"""
pipeline.py
End-to-end orchestrator. Single source of truth for the run order.

The full pipeline is: clean → engineer features → encode → segment →
train 7 models → evaluate → SHAP → RSI → recommendations → save.

Both main.py and run_pipeline.py call run_full_pipeline(); the
difference is purely the print format.
"""
import os

import pandas as pd

from src.config import DATA_RAW, DATA_PROCESSED_DIR, FIGURES_DIR
from src.data_cleaning       import load_and_clean, get_summary
from src.feature_engineering import engineer_features
from src.utils               import save_processed, encode_categoricals
from src.clustering          import segment_customers, compute_rsi, get_recommendation
from src.modeling            import train_all_models, evaluate_model, GEO_MAP, GEN_MAP
from src.explainability      import run_shap


def _generate_recommendations(df, model, scaler, feature_cols, decision_threshold):
    """
    Re-implementation of the deleted src/recommendation_engine.generate_recommendations
    body. Produces ChurnProbability, PredictedChurn, RevenueAtRisk, and
    Recommendation columns. PredictedChurn uses the tuned threshold from
    the model bundle, not the default 0.5.
    """
    dfc = df.copy()
    dfc['Geography_enc'] = dfc['Geography'].map(GEO_MAP).fillna(0).astype(int)
    dfc['Gender_enc']    = dfc['Gender'].map(GEN_MAP).fillna(0).astype(int)

    X = dfc[feature_cols].fillna(0)
    X_sc = scaler.transform(X)
    probas = model.predict_proba(X_sc)[:, 1]

    df['ChurnProbability'] = probas.round(4)
    df['PredictedChurn']   = (probas >= decision_threshold).astype(int)
    # Expected revenue at risk: balance * churn probability. A customer
    # with €100k and 0.50 churn probability contributes €50k to the
    # bank's exposure — useful for prioritising retention effort.
    df['RevenueAtRisk']    = (df['Balance'].fillna(0) * df['ChurnProbability']).round(2)
    df['Recommendation']   = df.apply(get_recommendation, axis=1)
    return df


def run_full_pipeline():
    """Execute the full analytics pipeline end-to-end. Returns the enriched DataFrame."""

    # ── 1. Cleaning ───────────────────────────────────────────────────────
    df = load_and_clean(str(DATA_RAW))
    s = get_summary(df)
    print(f"         {s['rows']:,} rows · {s['columns']} cols · churn {s['churn_rate']}%")

    # ── 2. Feature engineering ────────────────────────────────────────────
    df = engineer_features(df)
    df = encode_categoricals(df)
    save_processed(df, 'cleaned_dataset.csv')
    save_processed(df, 'features_dataset.csv')
    print(f"         {len(_get_feature_cols())} ML features ready")

    # ── 3. Segmentation ───────────────────────────────────────────────────
    df, seg_summary, X_sc, _km = segment_customers(df)
    save_processed(seg_summary, 'segment_summary.csv')

    # ── 4. Train 7 models + tune threshold on val ─────────────────────────
    model, scaler, feature_cols, decision_threshold = train_all_models(df)
    # Evaluate; figures land in data/processed/figures/ via FIGURES_DIR.
    metrics = evaluate_model(model, scaler, df, feature_cols,
                             output_dir=str(FIGURES_DIR),
                             decision_threshold=decision_threshold)

    # ── 5. SHAP explainability ────────────────────────────────────────────
    run_shap(model, scaler, df, feature_cols)

    # ── 6. RSI ────────────────────────────────────────────────────────────
    df = compute_rsi(df)
    rsi_dist = df['RSICategory'].value_counts()
    for cat, count in rsi_dist.items():
        print(f"  {cat}: {count:,} customers")

    # ── 7. Recommendations + final outputs ────────────────────────────────
    df = _generate_recommendations(df, model, scaler, feature_cols, decision_threshold)
    save_processed(df, 'final_segmented_dataset.csv')

    seg_insights = seg_summary.copy()
    save_processed(seg_insights, 'segment_insights.csv')

    # Enriched dataset (was customers_enriched.csv in the old run_pipeline.py)
    save_processed(df, 'customers_enriched.csv')

    # Recommendations distribution
    rec_dist = df['Recommendation'].value_counts()
    for rec, count in rec_dist.items():
        print(f"  {rec}: {count:,} customers")

    # Revenue at risk headline
    total_rar = df['RevenueAtRisk'].sum()
    print(f"\n  Revenue at risk (Σ Balance × P(churn)): €{total_rar:,.0f}")

    print(f"\n  Best Model : {type(model).__name__}")
    print(f"  ROC-AUC    : {metrics['roc_auc']*100:.2f}%")
    print(f"  Recall     : {metrics['recall']*100:.2f}%  @ threshold {decision_threshold:.3f}")
    print(f"  Outputs    : {DATA_PROCESSED_DIR}")
    print(f"  Model      : models/{os.environ.get('BEST_MODEL_NAME', 'best_model.pkl')}")
    print(f"\n  Launch dashboard: streamlit run streamlit_app/app.py\n")

    # ── 8. Refresh notebooks so output cells match this run ──────────────
    # Opt-out with SKIP_NOTEBOOKS=1 if nbclient / nbformat is not installed.
    if not os.environ.get('SKIP_NOTEBOOKS'):
        try:
            from notebooks.run_all import _main  # type: ignore
            _main()
        except Exception as e:
            print(f"  [notebooks] refresh skipped: {e}")

    return df


def _get_feature_cols():
    """Lazy import to avoid pulling feature_engineering at module load."""
    from src.feature_engineering import get_feature_list
    return get_feature_list()
