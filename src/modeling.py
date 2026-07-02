"""
modeling.py
Canonical home for model training and evaluation.

Consolidates the previously separate train_model.py and evaluate_model.py.
Bodies are preserved verbatim — only the module location has changed.
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve,
)

from src.config import (
    MODELS_DIR, DATA_PROCESSED_DIR, FIGURES_DIR, DEFAULT_MODEL_NAME,
    RANDOM_STATE, DECISION_THRESHOLD, CATBOOST_AUTO_CLASS_WEIGHTS, TEST_SIZE, VAL_SIZE,
)


# ── Shared constants ──────────────────────────────────────────────────────
FEATURE_COLS = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'BalanceSalaryRatio', 'ProductsPerTenure', 'EngagementScore',
    'WealthScore', 'RelationshipStrength'
]

GEO_MAP = {'France': 0, 'Spain': 1, 'Germany': 2}
GEN_MAP = {'Female': 0, 'Male': 1}


# ── Optional imports guarded so the module imports cleanly even if one
#     of the gradient-boosting libraries isn't installed in the env.
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

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


# ══════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════

def prepare_X_y(df):
    """Return feature matrix, target, and the full feature column list (incl. encodings)."""
    dfc = df.copy()
    dfc['Geography_enc'] = dfc['Geography'].map(GEO_MAP).fillna(0).astype(int)
    dfc['Gender_enc']    = dfc['Gender'].map(GEN_MAP).fillna(0).astype(int)
    feature_cols = FEATURE_COLS + ['Geography_enc', 'Gender_enc']
    X = dfc[feature_cols].fillna(0)
    y = dfc['Exited']
    return X, y, feature_cols


def _split_train_val_test(X, y):
    """
    60/20/20 stratified split. Test set is held out for final reporting;
    val set is used for threshold tuning so the tuned threshold doesn't
    leak into the test metrics.
    """
    # First carve out the test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    # Then carve val out of the remaining trainval
    # val fraction of remaining = VAL_SIZE / (1 - TEST_SIZE)
    val_frac_of_remaining = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac_of_remaining,
        random_state=RANDOM_STATE, stratify=y_trainval
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _tune_threshold(y_val, proba_val, target_recall: float = 0.70) -> float:
    """
    Pick the lowest threshold on the validation set that hits
    >= target_recall. Falls back to DECISION_THRESHOLD from config
    if no threshold in the candidate grid reaches the target.
    """
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    # precision_recall_curve returns one fewer threshold than (prec, rec);
    # pair them up by truncating.
    candidates = []
    for p, r, t in zip(prec[:-1], rec[:-1], thr):
        if r >= target_recall:
            candidates.append((p, r, float(t)))
    if not candidates:
        return DECISION_THRESHOLD
    # Among thresholds that hit target recall, pick the one with the
    # highest precision (the most precise threshold that still meets recall).
    candidates.sort(key=lambda x: (-x[0], x[2]))
    return candidates[0][2]


def train_all_models(df: pd.DataFrame, model_dir: str = None, output_dir: str = None,
                     target_recall: float = 0.70):
    """
    Train 7 ML models, select the best by ROC-AUC, persist via joblib.
    Uses a 60/20/20 train/val/test split with class-weighted CatBoost.
    Tunes the decision threshold on the val set to hit target_recall
    and saves the chosen threshold in the model bundle.

    Returns (best_model, scaler, feature_cols, decision_threshold).
    """
    model_dir = model_dir or str(MODELS_DIR)
    output_dir = output_dir or str(DATA_PROCESSED_DIR)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    X, y, feature_cols = prepare_X_y(df)

    X_train, X_val, X_test, y_train, y_val, y_test = _split_train_val_test(X, y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # SMOTE for class imbalance (fit only on the training fold to avoid
    # synthesising val/test rows into the decision boundary)
    if HAS_SMOTE:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = sm.fit_resample(X_train_sc, y_train)
    else:
        X_res, y_res = X_train_sc, y_train

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        'Extra Trees':         ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE,
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        )
    if HAS_LGB:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1,
            class_weight='balanced',
        )
    if HAS_CAT:
        models['CatBoost'] = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            random_seed=RANDOM_STATE, verbose=0,
            auto_class_weights=CATBOOST_AUTO_CLASS_WEIGHTS,
        )

    results = []
    best_auc = -1.0
    best_model = None
    best_name = ''
    best_proba_test = None
    best_proba_val  = None

    print(f"  [train_model] Training {len(models)} models...")

    for name, clf in models.items():
        clf.fit(X_res, y_res)
        proba_val  = clf.predict_proba(X_val_sc)[:, 1]
        proba_test = clf.predict_proba(X_test_sc)[:, 1]
        auc = roc_auc_score(y_test, proba_test)
        results.append({'Model': name, 'ROC_AUC': round(auc, 4)})
        print(f"    {name:25s}: ROC-AUC = {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_model = clf
            best_name = name
            best_proba_test = proba_test
            best_proba_val  = proba_val

    # Tune decision threshold on the val set for the *best* model.
    threshold = _tune_threshold(y_val, best_proba_val, target_recall=target_recall)

    print(f"\n  [train_model] Best model: {best_name} (ROC-AUC = {best_auc:.4f})")
    print(f"  [train_model] Decision threshold (tuned on val): {threshold:.3f}  "
          f"(target recall ≥ {target_recall:.2f})")

    # Persist model bundle — include the tuned threshold so inference
    # and the dashboard use the same operating point.
    bundle_path = os.path.join(model_dir, DEFAULT_MODEL_NAME)
    joblib.dump(
        {
            'model': best_model,
            'scaler': scaler,
            'features': feature_cols,
            'decision_threshold': float(threshold),
        },
        bundle_path
    )

    # Persist comparison table
    results_df = pd.DataFrame(results).sort_values('ROC_AUC', ascending=False)
    results_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)

    return best_model, scaler, feature_cols, float(threshold)


# ══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, scaler, df, feature_cols, output_dir: str = None,
                   decision_threshold: float = None):
    """
    Compute the standard metric set, write a model_comparison.csv if missing,
    and emit confusion-matrix, ROC, feature-importance, and engagement/product
    figures under output_dir/.

    Uses the held-out test split (60/20/20) and reports metrics at the
    *tuned* decision_threshold, not at the default 0.5.
    """
    output_dir = output_dir or str(FIGURES_DIR)
    threshold = DECISION_THRESHOLD if decision_threshold is None else decision_threshold
    os.makedirs(output_dir, exist_ok=True)

    dfc = df.copy()
    dfc['Geography_enc'] = dfc['Geography'].map(GEO_MAP).fillna(0).astype(int)
    dfc['Gender_enc']    = dfc['Gender'].map(GEN_MAP).fillna(0).astype(int)

    X = dfc[feature_cols].fillna(0)
    y = dfc['Exited']

    # Reproduce the same train/val/test split used at training time so the
    # metrics below are computed on the *same* held-out test set.
    _, _, X_test, _, _, y_test = _split_train_val_test(X, y)
    X_test_sc = scaler.transform(X_test)

    y_proba = model.predict_proba(X_test_sc)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    metrics = {
        'accuracy':   accuracy_score(y_test, y_pred),
        'precision':  precision_score(y_test, y_pred, zero_division=0),
        'recall':     recall_score(y_test, y_pred),
        'f1':         f1_score(y_test, y_pred),
        'roc_auc':    roc_auc_score(y_test, y_proba),
        'pr_auc':     average_precision_score(y_test, y_proba),
        'decision_threshold': float(threshold),
    }

    print(f"  [evaluate] Decision threshold : {threshold:.3f}")
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
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── ROC Curve ────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#3266ad', lw=2, label=f'ROC-AUC = {metrics["roc_auc"]:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random baseline')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#3266ad')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curve — {type(model).__name__}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Feature Importance ───────────────────────────────────────────────
    if hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        # write to processed (CSV consumer expects it there)
        imp.to_csv(os.path.join(DATA_PROCESSED_DIR, 'feature_importance.csv'), index=False)

        colors = ['#c0392b' if i >= len(imp) - 5 else '#3266ad' for i in range(len(imp))]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp['Feature'], imp['Importance'], color=colors, height=0.65)
        ax.set_xlabel('Feature Importance', fontsize=11)
        ax.set_title('Feature Importance — Top Churn Drivers', fontsize=13, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
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
    plt.savefig(os.path.join(output_dir, 'engagement_vs_churn.png'), dpi=150, bbox_inches='tight')
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
    plt.savefig(os.path.join(output_dir, 'products_vs_churn.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return metrics
