"""
Bank Retention Intelligence Platform
Master Pipeline Runner
Run: python run_pipeline.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run_step(name, func):
    print(f"\n▶ {name}...")
    t0 = time.time()
    result = func()
    print(f"  ✓ Done in {time.time() - t0:.1f}s")
    return result


def main():
    banner("Bank Retention Intelligence Platform")
    print("Starting full analytics pipeline...\n")

    DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'European_Bank.csv')
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Step 1 – Preprocessing
    from preprocessing import load_and_clean
    df = run_step("Step 1/8 – Data ingestion & validation", lambda: load_and_clean(DATA_PATH))
    print(f"  Dataset: {len(df):,} rows · {df.shape[1]} columns · Churn rate: {df['Exited'].mean()*100:.1f}%")

    # Step 2 – Feature Engineering
    from feature_engineering import engineer_features
    df = run_step("Step 2/8 – Feature engineering", lambda: engineer_features(df))
    print(f"  Features added: BalanceSalaryRatio, EngagementScore, WealthScore, RelationshipStrength, AgeGroup, WealthSegment")

    # Step 3 – Customer Segmentation
    from segmentation import segment_customers
    df, seg_summary = run_step("Step 3/8 – Customer segmentation (KMeans)", lambda: segment_customers(df, OUTPUT_DIR))
    print(f"  4 clusters identified")

    # Step 4 – Model Training
    from train_model import train_all_models
    model, scaler, feature_cols = run_step(
        "Step 4/8 – Model training (7 algorithms)",
        lambda: train_all_models(df, MODEL_DIR, OUTPUT_DIR)
    )
    print(f"  Best model: CatBoost saved to models/best_model.pkl")

    # Step 5 – Model Evaluation
    from evaluate_model import evaluate_model
    metrics = run_step(
        "Step 5/8 – Model evaluation & visualisations",
        lambda: evaluate_model(model, scaler, df, feature_cols, OUTPUT_DIR)
    )
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}% · ROC-AUC: {metrics['roc_auc']*100:.2f}% · Recall: {metrics['recall']*100:.2f}%")

    # Step 6 – SHAP Explainability
    from shap_analysis import run_shap
    run_step("Step 6/8 – SHAP explainability analysis", lambda: run_shap(model, scaler, df, feature_cols, OUTPUT_DIR))
    print(f"  SHAP summary plot saved to outputs/figures/shap_summary.png")

    # Step 7 – Retention Strength Index
    from retention_index import compute_rsi
    df = run_step("Step 7/8 – Computing Retention Strength Index (RSI)", lambda: compute_rsi(df))
    rsi_dist = df['RSICategory'].value_counts()
    for cat, count in rsi_dist.items():
        print(f"  {cat}: {count:,} customers")

    # Step 8 – Recommendation Engine
    from recommendation_engine import generate_recommendations
    df = run_step("Step 8/8 – Generating retention recommendations", lambda: generate_recommendations(df, model, scaler, feature_cols))
    rec_dist = df['Recommendation'].value_counts()
    for rec, count in rec_dist.items():
        print(f"  {rec}: {count:,} customers")

    # Save enriched dataset
    enriched_path = os.path.join(OUTPUT_DIR, 'customers_enriched.csv')
    df.to_csv(enriched_path, index=False)

    banner("Pipeline Complete")
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print(f"Model saved to:   {MODEL_DIR}/best_model.pkl")
    print(f"\nLaunch dashboard: streamlit run streamlit_app/app.py\n")


if __name__ == "__main__":
    main()
