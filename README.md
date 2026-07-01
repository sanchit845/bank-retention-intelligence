# 🏦 Bank Retention Intelligence Platform
**Customer Engagement & Product Utilisation Analytics for Retention Strategy**

---

## Project Structure

```
bank-retention-intelligence/
│
├── data/
│   ├── raw/
│   │   └── European_Bank.csv          ← Raw dataset (10,000 customers)
│   └── processed/
│       ├── cleaned_dataset.csv        ← After cleaning (notebook 01)
│       ├── features_dataset.csv       ← After feature engineering (notebook 02)
│       ├── model_comparison.csv       ← All 7 model metrics (notebook 03)
│       ├── feature_importance.csv     ← Feature importance scores
│       ├── final_segmented_dataset.csv← Full enriched dataset (notebook 04)
│       └── segment_insights.csv       ← Cluster profiles
│
├── models/
│   └── best_model.pkl                 ← Trained CatBoost + scaler + features
│
├── notebooks/
│   ├── 01_eda.ipynb                   ← Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb         ← Feature Engineering
│   ├── 03_clustering.ipynb            ← Model Training & Evaluation
│   └── 04_business_insights.ipynb     ← Segmentation, RSI, Strategy
│
├── reports/
│   └── screenshots/                   ← All saved charts (PNG)
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py               ← Load, validate, clean
│   ├── feature_engineering.py         ← 8 engineered features
│   ├── clustering.py                  ← KMeans, RSI, recommendations
│   ├── visualization.py               ← All reusable chart functions
│   └── utils.py                       ← File I/O, model save/load, helpers
│
├── streamlit_app/
│   └── app.py                         ← 7-page interactive dashboard
│
├── main.py                            ← One-command full pipeline runner
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run notebooks in order
```
notebooks/01_eda.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_clustering.ipynb
notebooks/04_business_insights.ipynb
```

### 3. Or run the full pipeline at once
```bash
python main.py
```

### 4. Launch the dashboard
```bash
streamlit run streamlit_app/app.py
```

---

## Key Findings

| Finding | Value |
|---|---|
| Overall churn rate | 20.4% |
| Active member churn | 14.3% |
| Inactive member churn | 26.9% |
| 1-product churn | 27.7% |
| 2-product churn | **7.6%** (optimal) |
| Germany churn | 32.4% |
| France/Spain churn | ~16% |
| Age 46–55 churn | **50.6%** (highest) |
| HV disengaged customers | 2,456 |
| Revenue at risk | **€104.7M** |

---

## Engineered Features

| Feature | Formula |
|---|---|
| BalanceSalaryRatio | Balance / EstimatedSalary |
| ProductsPerTenure | NumOfProducts / (Tenure + 1) |
| EngagementScore | 0.7 × IsActiveMember + 0.3 × HasCrCard |
| WealthScore | scaled Balance + Salary + CreditScore |
| RelationshipStrength | 0.4 × Engagement + 0.3 × Products + 0.3 × Tenure |
| AgeGroup | 18-25 / 26-35 / 36-45 / 46-55 / 56+ |
| WealthSegment | Low / Medium / High / Premium |
| HighValueDisengaged | Balance > median AND IsActiveMember = 0 |

---

## Model Performance (CatBoost)

| Metric | Score |
|---|---|
| Accuracy | 81.65% |
| ROC-AUC | **86.69%** |
| Recall | 73.96% |
| Precision | ~75% |

---

## Retention Strategies

1. **Immediate Outreach** — Customers with churn prob > 80%
2. **Cross-Sell Programme** — 5,084 single-product customers (27.7% → 7.6%)
3. **Reactivation Campaign** — 4,849 inactive members
4. **Germany Programme** — 32.4% regional churn anomaly
5. **Age 46–55 Advisory** — Highest-churn cohort (50.6%)
