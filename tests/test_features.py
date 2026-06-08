"""Unit tests for flight feature engineering."""
import unittest, sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','scripts'))

def make_df(n=80):
    np.random.seed(0)
    return pd.DataFrame({
        "fl_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "carrier": np.random.choice(["AA","DL","UA"], n),
        "origin":  np.random.choice(["ATL","LAX"], n),
        "dest":    np.random.choice(["JFK","SFO"], n),
        "dep_delay": np.random.normal(10, 25, n),
        "arr_delay": np.random.normal(8, 30, n),
        "distance":  np.random.uniform(200, 2500, n),
        "crs_dep_time": np.random.choice([600,800,1700,2000], n),
        "crs_arr_time": np.random.choice([800,1000,1900], n),
    })

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        from feature_engineering import engineer_features
        self.fn, self.df = engineer_features, make_df()

    def test_required_columns_present(self):
        r = self.fn(self.df)
        for c in ["day_of_week","month","is_weekend","dep_hour","is_peak_hour","is_delayed"]:
            self.assertIn(c, r.columns)

    def test_is_delayed_binary(self):
        self.assertTrue(self.fn(self.df)["is_delayed"].isin([0,1]).all())

    def test_delayed_above_15_min(self):
        r = self.fn(self.df)
        self.assertTrue((r.loc[r["arr_delay"]>15,"is_delayed"]==1).all())

    def test_day_of_week_0_to_6(self):
        self.assertTrue(self.fn(self.df)["day_of_week"].between(0,6).all())

    def test_no_input_mutation(self):
        cols = set(self.df.columns)
        self.fn(self.df)
        self.assertEqual(set(self.df.columns), cols)

class TestTrainModel(unittest.TestCase):
    def test_prepare_data_shapes(self):
        from feature_engineering import engineer_features
        from train_model import prepare_data
        X, y = prepare_data(engineer_features(make_df(200)))
        self.assertEqual(len(X), len(y))

    def test_no_nans_in_features(self):
        from feature_engineering import engineer_features
        from train_model import prepare_data
        X, y = prepare_data(engineer_features(make_df(200)))
        self.assertFalse(np.isnan(X).any())

if __name__ == "__main__": unittest.main()
