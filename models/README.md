# Models Directory

This directory stores trained ML models.

## After running `python run_pipeline.py`:

- `best_model.pkl` — Joblib bundle containing:
  - `model` — Best trained classifier (CatBoost by default)
  - `scaler` — Fitted StandardScaler
  - `features` — List of feature column names

## Loading the model in Python:

```python
import joblib

bundle = joblib.load('models/best_model.pkl')
model      = bundle['model']
scaler     = bundle['scaler']
features   = bundle['features']
```

## Expected performance (CatBoost):

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 81.65%  |
| ROC-AUC   | 86.69%  |
| Recall    | 73.96%  |
| Precision | ~75%    |
| F1        | ~0.78   |
