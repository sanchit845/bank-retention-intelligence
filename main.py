"""
main.py
Bank Retention Intelligence Platform — Master Runner

Runs the full analytics pipeline end-to-end:
  1. Data cleaning
  2. Feature engineering
  3. Customer segmentation
  4. Model training
  5. Model evaluation
  6. SHAP analysis
  7. RSI computation
  8. Recommendation engine

Usage:
    python main.py
"""

import os, sys, time, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_cleaning       import load_and_clean, get_summary
from src.feature_engineering import engineer_features, get_feature_list
from src.clustering          import segment_customers, compute_rsi, get_recommendation
from src.utils               import (save_processed, save_model, save_figure,
                                     encode_categoricals, get_path, print_section)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def run_step(name, fn):
    print(f"\n  ▶ {name}", end=" ... ", flush=True)
    t = time.time()
    result = fn()
    print(f"✓  ({time.time()-t:.1f}s)")
    return result


def main():
    print_section("Bank Retention Intelligence Platform — Full Pipeline")

    DATA_PATH = get_path('data', 'raw', 'European_Bank.csv')

    # Step 1 — Clean
    df = run_step("Step 1/8  Data cleaning",
                  lambda: load_and_clean(DATA_PATH))
    s = get_summary(df)
    print(f"         {s['rows']:,} rows · {s['columns']} cols · churn {s['churn_rate']}%")

    # Step 2 — Feature engineering
    df = run_step("Step 2/8  Feature engineering",
                  lambda: engineer_features(df))
    df = encode_categoricals(df)
    save_processed(df, 'cleaned_dataset.csv')

    # Step 3 — Feature save
    save_processed(df, 'features_dataset.csv')
    print(f"         {len(get_feature_list())} ML features ready")

    # Step 4 — Segmentation
    df, seg_summary, X_sc, _ = run_step("Step 4/8  KMeans segmentation",
                                         lambda: segment_customers(df))
    save_processed(seg_summary, 'segment_summary.csv')

    # Step 5 — Model training
    print_section("Step 5/8  Model Training")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
    from imblearn.over_sampling import SMOTE

    FEATURE_COLS = get_feature_list()
    X = df[FEATURE_COLS].fillna(0)
    y = df['Exited']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_tr_sc, y_tr)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'Extra Trees':         ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }
    for lib, cls, kwargs in [
        ('xgboost',  'XGBClassifier',  dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                                             use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('lightgbm', 'LGBMClassifier', dict(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)),
        ('catboost', 'CatBoostClassifier', dict(iterations=500, learning_rate=0.05, depth=6, random_seed=42, verbose=0)),
    ]:
        try:
            mod = __import__(lib)
            models[cls.replace('Classifier','').replace('LGBM','LightGBM').replace('XGB','XGBoost').replace('CatBoost','CatBoost')] = getattr(mod, cls)(**kwargs)
        except ImportError:
            pass

    results, best_auc, best_model, best_name = [], -1, None, ''
    for name, clf in models.items():
        print(f"    Training {name:25s} ...", end=' ', flush=True)
        clf.fit(X_res, y_res)
        proba = clf.predict_proba(X_te_sc)[:, 1]
        pred  = (proba > 0.5).astype(int)
        auc = roc_auc_score(y_te, proba)
        acc = accuracy_score(y_te, pred) * 100
        rec = recall_score(y_te, pred) * 100
        results.append({'Model': name, 'Accuracy': round(acc,2), 'ROC_AUC': round(auc*100,2), 'Recall': round(rec,2)})
        print(f"AUC={auc*100:.2f}%  Acc={acc:.2f}%  Recall={rec:.2f}%")
        if auc > best_auc:
            best_auc, best_model, best_name = auc, clf, name

    results_df = pd.DataFrame(results).sort_values('ROC_AUC', ascending=False)
    save_processed(results_df, 'model_comparison.csv')
    save_model(best_model, scaler, FEATURE_COLS)
    print(f"\n  Best model: {best_name}  AUC={best_auc*100:.2f}%")

    # Step 6 — RSI + Recommendations
    df = run_step("Step 6/8  RSI computation", lambda: compute_rsi(df))
    X_all_sc = scaler.transform(df[FEATURE_COLS].fillna(0))
    df['ChurnProbability'] = best_model.predict_proba(X_all_sc)[:, 1].round(4)
    df['Recommendation']   = df.apply(get_recommendation, axis=1)

    # Step 7 — Save final outputs
    save_processed(df, 'final_segmented_dataset.csv')
    seg_insights = seg_summary.copy()
    save_processed(seg_insights, 'segment_insights.csv')

    print_section("Pipeline Complete")
    print(f"  Best Model : {best_name}")
    print(f"  ROC-AUC    : {best_auc*100:.2f}%")
    print(f"  Outputs    : data/processed/")
    print(f"  Model      : models/best_model.pkl")
    print(f"\n  Launch dashboard: streamlit run streamlit_app/app.py\n")


if __name__ == '__main__':
    main()
