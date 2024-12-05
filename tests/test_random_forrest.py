import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from apple_stock_ml import SEED
from apple_stock_ml.src.random_forest import (
    create_sequences,
    train_random_forest,
    evaluate_model,
    save_model,
)


class TestRandomForest(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.X = pd.DataFrame({"feature1": range(15), "feature2": range(15, 30)})
        self.y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    def test_create_sequences(self):
        # Test sequence creation with sequence_length=3
        sequences, targets = create_sequences(self.X, self.y, sequence_length=3)

        # Check shapes
        self.assertEqual(
            sequences.shape, (12, 3, 2)
        )  # 12 sequences, length 3, 2 features
        self.assertEqual(targets.shape, (12,))  # 12 targets

        # Check first sequence
        expected_first_seq = self.X.iloc[0:3].values
        np.testing.assert_array_equal(sequences[0], expected_first_seq)

        # Check first target
        self.assertEqual(targets[0], self.y.iloc[3])

    def test_train_random_forest(self):
        # Test with default parameters
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)

        model = train_random_forest(X_train, y_train)

        self.assertIsInstance(model, RandomForestClassifier)
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.random_state, SEED)

    def test_evaluate_model(self):
        # Create a mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1])

        X_test = np.random.rand(4, 10)
        y_test = np.array([0, 1, 0, 1])

        with self.assertLogs() as captured:
            predictions = evaluate_model(mock_model, X_test, y_test, "Test")

        # Check if predictions were made
        np.testing.assert_array_equal(predictions, np.array([0, 1, 0, 1]))

        # Check if logging occurred
        self.assertTrue(
            any("Performance on Test Set" in msg for msg in captured.output)
        )

    def test_save_model(self):
        # Create a temporary directory for testing
        test_dir = Path("test_models")
        test_file = test_dir / "test_model.pkl"

        # Create a mock model
        mock_model = RandomForestClassifier()

        try:
            # Test saving the model
            save_model(mock_model, str(test_file))

            # Check if file exists
            self.assertTrue(test_file.exists())

            # Check if file is not empty
            self.assertGreater(test_file.stat().st_size, 0)

        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()
            if test_dir.exists():
                test_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
