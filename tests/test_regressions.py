"""
tests/test_regressions.py
Regression tests for the Bank Retention Intelligence Platform.

Catches the bugs that have already been fixed in this codebase, so they
don't reappear:
  1. predict_churn and predict_single must produce identical results
     for the same customer (engineer_features on a 1-row frame used to
     collapse the MinMax scalers to 0).
  2. The saved model bundle must include a decision_threshold and a
     non-empty feature list.
  3. The enriched dataset must carry every column the dashboard reads.
  4. The trained model must hit Recall >= 0.60 on the held-out test
     set at the tuned threshold (was 0.5479 before the threshold /
     class-weight fix).
  5. KMeans segmentation must produce exactly 4 clusters with the
     expected names.

Run directly:  python -m tests.test_regressions
Run with pytest: pytest tests/
"""
import os
import sys
import unittest

# Allow running this file from the project root without installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


REQUIRED_ENRICHED_COLUMNS = [
    'Exited', 'Geography', 'Gender', 'Age', 'Balance', 'IsActiveMember',
    'NumOfProducts', 'Tenure',
    'ChurnProbability', 'PredictedChurn', 'RevenueAtRisk', 'Recommendation',
    'RSI', 'RSICategory', 'Cluster',
]
EXPECTED_CLUSTER_NAMES = {'Young Active', 'Premium Loyal', 'Wealthy Disengaged', 'High Risk'}
TARGET_RECALL_FLOOR     = 0.60  # post-threshold / post-class-weight floor


class TestInferenceInvariant(unittest.TestCase):
    """The bug we already fixed: predict_churn and predict_single must agree."""

    @classmethod
    def setUpClass(cls):
        from src.inference import load_model
        cls.bundle = load_model()
        cls.model, cls.scaler, cls.features, cls.threshold = cls.bundle

    def test_bundle_has_four_elements(self):
        self.assertEqual(len(self.bundle), 4, "load_model must return 4-tuple")

    def test_decision_threshold_is_float_in_unit_interval(self):
        self.assertIsInstance(self.threshold, float)
        self.assertGreater(self.threshold, 0.0)
        self.assertLessEqual(self.threshold, 1.0)

    def test_feature_list_not_empty(self):
        self.assertGreater(len(self.features), 0)
        for required in ['Age', 'Balance', 'NumOfProducts', 'Geography_enc', 'Gender_enc']:
            self.assertIn(required, self.features)

    def test_predict_churn_equals_predict_single(self):
        """The exact invariant that broke last session."""
        from src.inference import predict_churn, predict_single
        customer = {
            'CreditScore': 580, 'Geography': 'Germany', 'Gender': 'Female', 'Age': 52,
            'Tenure': 2, 'Balance': 145000, 'NumOfProducts': 1, 'HasCrCard': 1,
            'IsActiveMember': 0, 'EstimatedSalary': 98000,
        }
        r1 = predict_churn(customer)
        r2 = predict_single(customer, self.model, self.scaler, self.features, self.threshold)
        # Probabilities are the fundamental output; everything else derives from it.
        self.assertEqual(r1['churn_probability'], r2['churn_probability'])
        self.assertEqual(r1['risk_level'],        r2['risk_level'])
        self.assertEqual(r1['rsi_category'],      r2['rsi_category'])
        self.assertEqual(r1['recommendation'],    r2['recommendation'])
        self.assertEqual(r1['predicted_churn'],  r2['predicted_churn'])

    def test_predict_churn_returns_revenue_at_risk(self):
        from src.inference import predict_churn
        customer = {
            'CreditScore': 700, 'Geography': 'France', 'Gender': 'Male', 'Age': 30,
            'Tenure': 5, 'Balance': 100000, 'NumOfProducts': 2, 'HasCrCard': 1,
            'IsActiveMember': 1, 'EstimatedSalary': 100000,
        }
        r = predict_churn(customer)
        self.assertIn('revenue_at_risk', r)
        self.assertGreaterEqual(r['revenue_at_risk'], 0.0)


class TestEnrichedDataset(unittest.TestCase):
    """The enriched CSV the dashboard reads must carry every required column."""

    ENRICHED_PATH = os.path.join('data', 'processed', 'customers_enriched.csv')

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        cls.df = pd.read_csv(cls.ENRICHED_PATH)

    def test_enriched_file_exists(self):
        self.assertTrue(os.path.exists(self.ENRICHED_PATH),
                        f"Run `python main.py` to generate {self.ENRICHED_PATH}")

    def test_all_dashboard_columns_present(self):
        missing = [c for c in REQUIRED_ENRICHED_COLUMNS if c not in self.df.columns]
        self.assertEqual(missing, [], f"Missing columns: {missing}")

    def test_revenue_at_risk_non_negative(self):
        import pandas as pd
        self.assertTrue((self.df['RevenueAtRisk'].fillna(0) >= 0).all())

    def test_predicted_churn_is_binary(self):
        self.assertTrue(set(self.df['PredictedChurn'].unique()).issubset({0, 1}))

    def test_segment_count_is_four(self):
        import pandas as pd
        self.assertEqual(self.df['Cluster'].nunique(), 4)

    def test_segment_names_match_spec(self):
        actual = set(self.df['Cluster'].unique())
        self.assertEqual(actual, EXPECTED_CLUSTER_NAMES,
                         f"Cluster names drifted from spec: {actual}")


class TestModelPerformance(unittest.TestCase):
    """The model must clear a Recall floor on the held-out test set."""

    @classmethod
    def setUpClass(cls):
        from src.modeling import _split_train_val_test, prepare_X_y
        from src.inference import load_model
        from src.data_cleaning import load_and_clean
        from src.feature_engineering import engineer_features
        from src.utils import encode_categoricals
        import pandas as pd
        from sklearn.metrics import recall_score

        cls.model, cls.scaler, cls.features, cls.threshold = load_model()

        df = load_and_clean('data/raw/European_Bank.csv')
        df = engineer_features(df)
        df = encode_categoricals(df)
        X, y, _ = prepare_X_y(df)
        _, _, X_test, _, _, y_test = _split_train_val_test(X, y)
        X_test_sc = cls.scaler.transform(X_test)
        proba = cls.model.predict_proba(X_test_sc)[:, 1]
        y_pred = (proba >= cls.threshold).astype(int)
        cls.recall = recall_score(y_test, y_pred)

    def test_recall_above_floor(self):
        self.assertGreaterEqual(
            self.recall, TARGET_RECALL_FLOOR,
            f"Recall {self.recall:.3f} fell below floor {TARGET_RECALL_FLOOR:.2f}. "
            f"Was the threshold / class-weight change reverted?"
        )


def _print_summary():
    """Run with python -m tests.test_regressions — print a compact summary."""
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestInferenceInvariant, TestEnrichedDataset, TestModelPerformance]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    _print_summary()
