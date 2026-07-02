"""
config.py
Single source of truth for project paths and tunables.

All modules that need a path or a constant should import from here,
not hard-code it. This makes the layout portable and refactor-safe.
"""
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT       = Path(__file__).resolve().parent.parent
DATA_RAW           = PROJECT_ROOT / 'data' / 'raw' / 'European_Bank.csv'
DATA_PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
# Figures live alongside the CSVs that produced them — single artifact tree.
FIGURES_DIR        = DATA_PROCESSED_DIR / 'figures'
MODELS_DIR         = PROJECT_ROOT / 'models'

DEFAULT_MODEL_NAME = 'best_model.pkl'


# ── Tunables ──────────────────────────────────────────────────────────────
RANDOM_STATE                 = 42
HIGH_VALUE_BALANCE_THRESHOLD = 97199   # median balance — replaces magic numbers in 4 files
CHURN_HIGH_THRESHOLD         = 0.80    # immediate-outreach cutoff
CHURN_MODERATE_THRESHOLD     = 0.50    # moderate-risk cutoff
SAMPLE_SHAP_SIZE             = 1000    # SHAP background / sample size

# Decision threshold for flagging a customer as a churner. Lower ⇒ higher
# Recall (catches more churners, more false-positive outreach calls).
# 0.30 is the right operating point for a retention use case — the cost
# of a retention call to a non-churner is far less than losing a churner.
DECISION_THRESHOLD           = 0.30

# CatBoost class-weight setting. 'Balanced' up-weights the minority (churn)
# class during training so the resulting probabilities are not biased
# toward the majority (retained) class.
CATBOOST_AUTO_CLASS_WEIGHTS  = 'Balanced'

# Train / val / test split ratios. Threshold is tuned on val, final metrics
# reported on test, so threshold-tuning does not leak into the test set.
TEST_SIZE                    = 0.20
VAL_SIZE                     = 0.20   # fraction of the *original* dataset
