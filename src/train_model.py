"""
train_model.py
Trains 7 ML models, selects the best by ROC-AUC, saves it with joblib.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


FEATURE_COLS = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'BalanceSalaryRatio', 'ProductsPerTenure', 'EngagementScore',
    'WealthScore', 'RelationshipStrength'
]

GEO_MAP = {'France': 0, 'Spain': 1, 'Germany': 2}
GEN_MAP = {'Female': 0, 'Male': 1}


def prepare_X_y(df):
    dfc = df.copy()
    dfc['Geography_enc'] = dfc['Geography'].map(GEO_MAP).fillna(0).astype(int)
    dfc['Gender_enc'] = dfc['Gender'].map(GEN_MAP).fillna(0).astype(int)
    feature_cols = FEATURE_COLS + ['Geography_enc', 'Gender_enc']
    X = dfc[feature_cols].fillna(0)
    y = dfc['Exited']
    return X, y, feature_cols


def train_all_models(df: pd.DataFrame, model_dir: str, output_dir: str):
    X, y, feature_cols = prepare_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Apply SMOTE to handle class imbalance
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_sc, y_train)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'Extra Trees': ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            use_label_encoder=False, eval_metric='logloss', random_state=42
        )
    if HAS_LGB:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1
        )
    if HAS_CAT:
        models['CatBoost'] = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            random_seed=42, verbose=0
        )

    results = []
    best_auc = -1
    best_model = None
    best_name = ''

    print(f"  [train_model] Training {len(models)} models...")

    for name, clf in models.items():
        clf.fit(X_res, y_res)
        proba = clf.predict_proba(X_test_sc)[:, 1]
        auc = roc_auc_score(y_test, proba)
        results.append({'Model': name, 'ROC_AUC': round(auc, 4)})
        print(f"    {name:25s}: ROC-AUC = {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_model = clf
            best_name = name

    print(f"\n  [train_model] Best model: {best_name} (ROC-AUC = {best_auc:.4f})")

    # Save model + scaler + feature list
    joblib.dump(
        {'model': best_model, 'scaler': scaler, 'features': feature_cols},
        os.path.join(model_dir, 'best_model.pkl')
    )

    # Save comparison table
    results_df = pd.DataFrame(results).sort_values('ROC_AUC', ascending=False)
    results_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)

    return best_model, scaler, feature_cols
