# Outputs Directory

This directory is populated when you run `python run_pipeline.py`.

## Contents after pipeline run:

- `model_comparison.csv` — Performance metrics for all 7 trained models
- `feature_importance.csv` — Feature importance scores from the best model
- `segment_summary.csv` — KMeans cluster characteristics
- `shap_feature_importance.csv` — Mean absolute SHAP values per feature
- `customers_enriched.csv` — Full dataset with ChurnProbability, RSI, Recommendation
- `figures/` — All charts and visualisations
  - `churn_distribution.png`
  - `engagement_vs_churn.png`
  - `products_vs_churn.png`
  - `customer_segments.png`
  - `roc_curve.png`
  - `confusion_matrix.png`
  - `feature_importance.png`
  - `shap_summary.png`
  - `rsi_analysis.png`
  - `high_value_disengaged.png`
