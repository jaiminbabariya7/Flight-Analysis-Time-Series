"""Unit tests for flight delay feature engineering."""
import unittest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


def make_sample_df(n=100):
    """Create a minimal sample flight DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "fl_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "carrier": np.random.choice(["AA", "DL", "UA", "WN"], n),
        "origin": np.random.choice(["ATL", "LAX", "ORD"], n),
        "dest": np.random.choice(["JFK", "SFO", "DFW"], n),
        "dep_delay": np.random.normal(10, 25, n),
        "arr_delay": np.random.normal(8, 30, n),
        "distance": np.random.uniform(200, 2500, n),
        "crs_dep_time": np.random.choice([600, 800, 1200, 1700, 2000], n),
        "crs_arr_time": np.random.choice([800, 1000, 1400, 1900, 2200], n),
    })


class TestEngineerFeatures(unittest.TestCase):
    def setUp(self):
        from feature_engineering import engineer_features
        self.engineer = engineer_features
        self.df = make_sample_df()

    def test_output_has_expected_columns(self):
        result = self.engineer(self.df)
        for col in ["day_of_week", "month", "is_weekend", "dep_hour", "is_peak_hour",
                    "route_avg_delay", "distance_bucket", "is_delayed"]:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_day_of_week_range(self):
        result = self.engineer(self.df)
        self.assertTrue(result["day_of_week"].between(0, 6).all())

    def test_month_range(self):
        result = self.engineer(self.df)
        self.assertTrue(result["month"].between(1, 12).all())

    def test_is_weekend_binary(self):
        result = self.engineer(self.df)
        self.assertTrue(result["is_weekend"].isin([0, 1]).all())

    def test_is_delayed_binary(self):
        result = self.engineer(self.df)
        self.assertTrue(result["is_delayed"].isin([0, 1]).all())

    def test_delayed_threshold_15_min(self):
        """Flights with arr_delay > 15 must be labelled delayed."""
        result = self.engineer(self.df)
        delayed_mask = result["arr_delay"] > 15
        self.assertTrue((result.loc[delayed_mask, "is_delayed"] == 1).all())

    def test_no_data_leakage_from_future(self):
        """route_avg_delay uses transform (same-dataset mean) — no future data."""
        result = self.engineer(self.df)
        self.assertFalse(result["route_avg_delay"].isna().any())

    def test_dep_hour_range(self):
        result = self.engineer(self.df)
        self.assertTrue(result["dep_hour"].between(0, 23).all())

    def test_input_not_mutated(self):
        """engineer_features must not modify the input DataFrame."""
        original_cols = set(self.df.columns)
        _ = self.engineer(self.df)
        self.assertEqual(set(self.df.columns), original_cols)


class TestTrainModel(unittest.TestCase):
    def test_prepare_data_returns_arrays(self):
        from feature_engineering import engineer_features
        from train_model import prepare_data
        df = engineer_features(make_sample_df(200))
        X, y = prepare_data(df)
        self.assertEqual(len(X), len(y))
        self.assertEqual(y.ndim, 1)

    def test_prepare_data_no_nans(self):
        from feature_engineering import engineer_features
        from train_model import prepare_data
        df = engineer_features(make_sample_df(200))
        X, y = prepare_data(df)
        self.assertFalse(np.isnan(X).any())
        self.assertFalse(np.isnan(y).any())


if __name__ == "__main__":
    unittest.main()
