"""
utils.py
Shared utility functions used across notebooks and main.py.

`get_path` is now a thin wrapper around src.config so every module
resolves paths the same way. The legacy `predict_single` was moved
to src.inference — keep that import path going forward.
"""
import os
import sys
import pandas as pd
import joblib

# Force UTF-8 stdout/stderr on Windows so the box-drawing banner chars
# below don't crash on cp1252. No-op on macOS/Linux.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

from src.config import (
    PROJECT_ROOT,
    DATA_RAW,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    DEFAULT_MODEL_NAME,
)


# ── Paths ─────────────────────────────────────────────────────────────────

def get_project_root() -> str:
    """Return absolute path to the project root directory."""
    return str(PROJECT_ROOT)


def get_path(*parts) -> str:
    """
    Build an absolute path relative to the project root.

    Back-compat shim — for parts that map to a known config key the
    canonical Path is returned; otherwise a join on PROJECT_ROOT is used.
    """
    if parts == ('data', 'raw', 'European_Bank.csv'):
        return str(DATA_RAW)
    if parts == ('data', 'processed'):
        return str(DATA_PROCESSED_DIR)
    if parts == ('models',):
        return str(MODELS_DIR)
    if parts == ('models', DEFAULT_MODEL_NAME):
        return str(MODELS_DIR / DEFAULT_MODEL_NAME)
    return str(PROJECT_ROOT.joinpath(*parts))


# ── Model I/O ─────────────────────────────────────────────────────────────

def save_model(model, scaler, feature_cols: list, filename: str = None):
    """Save model bundle (model + scaler + features) to models/."""
    filename = filename or DEFAULT_MODEL_NAME
    path = MODELS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler, 'features': feature_cols}, path)
    print(f"  Model saved → {path}")
    return str(path)


def load_model(filename: str = None):
    """Load saved model bundle. Returns (model, scaler, feature_cols, decision_threshold)."""
    filename = filename or DEFAULT_MODEL_NAME
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            f"Run the pipeline (python main.py) first to train and save the model."
        )
    bundle = joblib.load(path)
    threshold = bundle.get('decision_threshold')
    if threshold is None:
        from src.config import DECISION_THRESHOLD as default_threshold
        threshold = default_threshold
    return bundle['model'], bundle['scaler'], bundle['features'], float(threshold)


# ── Data I/O ──────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame, filename: str):
    """Save a DataFrame to data/processed/."""
    path = DATA_PROCESSED_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved → {path}  ({len(df):,} rows)")
    return str(path)


def load_processed(filename: str) -> pd.DataFrame:
    """Load a DataFrame from data/processed/."""
    path = DATA_PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def save_figure(fig, filename: str, dpi: int = 150):
    """
    Save a matplotlib figure to data/processed/figures/ (the canonical
    home for every chart the pipeline produces). Back-compat: a `folder`
    kwarg is accepted but ignored, so older notebook code keeps working
    after the outputs/ directory was removed.
    """
    path = FIGURES_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"  Figure saved → {path}")
    return str(path)


# ── Encoding helpers ──────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Add Geography_enc and Gender_enc columns."""
    GEO_MAP = {'France': 0, 'Spain': 1, 'Germany': 2}
    GEN_MAP = {'Female': 0, 'Male': 1}
    df = df.copy()
    df['Geography_enc'] = df['Geography'].map(GEO_MAP).fillna(0).astype(int)
    df['Gender_enc']    = df['Gender'].map(GEN_MAP).fillna(0).astype(int)
    return df


# ── Display helpers ───────────────────────────────────────────────────────

def print_section(title: str, width: int = 60):
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def print_metrics(metrics: dict):
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v*100:.2f}%")
        else:
            print(f"  {k:20s}: {v}")
