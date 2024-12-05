import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path
import pickle
from apple_stock_ml.src.data_preprocessing import (
    load_stock_data,
    calculate_daily_returns,
    create_extreme_events,
    prepare_features,
    split_time_series_data,
)


class TestDataPreprocessing(unittest.TestCase):

    @patch("apple_stock_ml.src.data_preprocessing.yf.download")
    def test_load_stock_data(self, mock_download):
        # Mock the data returned by yfinance
        mock_data = pd.DataFrame(
            {
                "Adj Close": [100, 101, 102],
                "Open": [99, 100, 101],
                "High": [101, 102, 103],
                "Low": [98, 99, 100],
                "Close": [100, 101, 102],
                "Volume": [1000, 1100, 1200],
            }
        )
        mock_download.return_value = mock_data

        # Test loading data without cache
        df = load_stock_data("2022-01-01", "2022-01-03", "AAPL")
        pd.testing.assert_frame_equal(df, mock_data)

    def test_calculate_daily_returns(self):
        df = pd.DataFrame({"Adj Close": [100, 102, 101]})
        df = calculate_daily_returns(df)
        expected_returns = [None, 2.0, -0.9803921568627451]
        self.assertEqual(df["Daily_Return"].tolist(), expected_returns)

    def test_create_extreme_events(self):
        df = pd.DataFrame({"Daily_Return": [1.0, 3.0, -2.5]})
        df = create_extreme_events(df, threshold=2.0)
        expected_events = [0, 1, 0]
        self.assertEqual(df["Extreme_Event"].tolist(), expected_events)

    def test_prepare_features(self):
        df = pd.DataFrame(
            {
                "Open": [99, 100, 101],
                "High": [101, 102, 103],
                "Low": [98, 99, 100],
                "Close": [100, 101, 102],
                "Volume": [1000, 1100, 1200],
                "Daily_Return": [None, 2.0, -0.9803921568627451],
                "Extreme_Event": [0, 1, 0],
            }
        )
        X, y = prepare_features(df)
        self.assertEqual(X.shape, (2, 6))
        self.assertEqual(y.shape, (2,))

    def test_split_time_series_data(self):
        X = pd.DataFrame({"feature": range(10)})
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        X_train, X_val, X_test, y_train, y_val, y_test = split_time_series_data(X, y)
        self.assertEqual(len(X_train), 7)
        self.assertEqual(len(X_val), 1)
        self.assertEqual(len(X_test), 2)


if __name__ == "__main__":
    unittest.main()
