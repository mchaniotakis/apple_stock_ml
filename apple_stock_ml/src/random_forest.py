"""Random Forest model for predicting extreme events in Apple stock prices."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import argparse
from pathlib import Path
import pickle
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from apple_stock_ml.src.utils.logger import setup_logger
from apple_stock_ml import set_seeds
from apple_stock_ml.src.data_preprocessing import (
    get_data,
)
from sklearn.metrics import f1_score, make_scorer
from apple_stock_ml.src.utils.visualizer import ModelVisualizer
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm
import time

set_seeds()
# Set up logger
logger = setup_logger("random_forest", "logs/random_forest.log")


# def create_sequences(X, y, sequence_length=10):
#     """Create sequences of specified length from the data."""
#     sequences = []
#     targets = []

#     for i in range(len(X) - sequence_length):
#         seq = X.iloc[i : i + sequence_length].values
#         target = y.iloc[i + sequence_length]
#         sequences.append(seq)
#         targets.append(target)

#     return np.array(sequences), np.array(targets)


def train_random_forest(
    X_train, y_train, X_val=None, y_val=None, optimize_hyperparams=True
):
    """Train Random Forest model with optional hyperparameter optimization."""
    logger.info("Training Random Forest model...")

    if optimize_hyperparams:
        param_grid = {
            "n_estimators": [200, 300, 400],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": ["balanced", "balanced_subsample"],
            "bootstrap": [True, False],
        }

        # Calculate total iterations for progress bar
        total_iterations = sum(len(v) for v in param_grid.values())
        logger.info(
            f"Starting hyperparameter optimization with {total_iterations} combinations"
        )

        f1_scorer = make_scorer(f1_score, average="weighted")

        model = RandomizedSearchCV(
            RandomForestClassifier(random_state=SEED),
            param_grid,
            n_iter=20,
            cv=3,
            n_jobs=-1,
            verbose=0,  # Set to 0 to avoid conflicting output
            scoring=f1_scorer,
        )
        logger.info("Fitting RandomizedSearchCV...")
        model.fit(X_train, y_train)

        # Log the best parameters and score
        logger.info(f"Best parameters found: {model.best_params_}")
        logger.info(f"Best cross-validation score: {model.best_score_:.4f}")

        # Use the best estimator
        model = model.best_estimator_

    else:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=1,  # Enable built-in verbosity
        )

    # Train in batches if dataset is large
    if len(X_train) > 100000:
        logger.info("Large dataset detected, training in batches...")
        train_model_in_batches(model, X_train, y_train, batch_size=500)
    else:
        start_time = time.time()
        logger.info("Starting model training...")
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

    return model


def train_model_in_batches(model, X_train, y_train, batch_size=50000):
    """Train model in batches to handle large datasets."""
    n_samples = len(X_train)
    n_batches = (n_samples - 1) // batch_size + 1

    with tqdm(total=n_batches, desc="Training Batches") as pbar:
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]

            batch_start_time = time.time()
            if i == 0:
                model.fit(X_batch, y_batch)
            else:
                for tree in model.estimators_:
                    tree.fit(X_batch, y_batch)

            batch_time = time.time() - batch_start_time
            pbar.set_postfix(
                {"batch_size": len(X_batch), "batch_time": f"{batch_time:.2f}s"}
            )
            pbar.update(1)


def evaluate_model(model, X, y, set_name="latest"):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Calculate confusion matrix
    cm = confusion_matrix(y, predictions)

    # Calculate sensitivity (True Positive Rate) and specificity (True Negative Rate)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Calculate classification report
    report = classification_report(y, predictions, output_dict=True)

    # Log results
    logger.info(f"\nPerformance on {set_name} Set:")
    logger.info("\nConfusion Matrix:")
    logger.info(cm)
    logger.info("\nClassification Report:")
    logger.info(classification_report(y, predictions))
    logger.info(f"Sensitivity (TPR): {sensitivity:.4f}")
    logger.info(f"Specificity (TNR): {specificity:.4f}")

    postive_to_negative_ratio = np.sum(y) / len(y)
    prediction_positive_to_negative_ratio = np.sum(predictions) / len(predictions)

    # Return comprehensive metrics
    metrics = {
        "predictions": predictions,
        "true_values": y,
        "probabilities": probabilities,
        "confusion_matrix": cm,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "postive_to_negative_ratio": postive_to_negative_ratio,
        "prediction_positive_to_negative_ratio": prediction_positive_to_negative_ratio,
        "classification_report": report,
    }

    return metrics


def load_model(filepath):
    """Load the trained model from disk."""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model


def save_model(model, filepath):
    """Save the trained model to disk."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {filepath}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate Random Forest model"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2015-01-01",
        help="Start date for stock data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-01-31",
        help="End date for stock data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--ticker", type=str, default="AAPL", help="Stock ticker symbol"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Threshold for extreme events (percentage)",
    )
    parser.add_argument(
        "--optimize",
        "-o",
        action="store_true",
        help="Perform hyperparameter optimization",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Directory for cached data",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="models/random_forest.pkl",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--use-smote", action="store_true", help="Apply SMOTE for class balancing"
    )
    return parser.parse_args()


def normalize_data(X_train, X_val, X_test):
    """Normalize the data using training set statistics, except for the last channel"""
    mean = X_train.mean()
    std = X_train.std()

    X_train_norm = (X_train - mean) / (std + 1e-8)
    X_val_norm = (X_val - mean) / (std + 1e-8)
    X_test_norm = (X_test - mean) / (std + 1e-8)

    return X_train_norm, X_val_norm, X_test_norm


def main(
    ticker: str,
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-01",
    threshold: float = 2.0,
    sequence_length: int = 10,
    use_smote: bool = False,
    optimize: bool = False,
    model_output: str = "models/random_forest.pkl",
):
    # we need to flatten the data for random forrest
    # features = ["Adj Close", "Close", "High", "Low", "Open", "Volume", "Daily_Return"]
    idx_to_keep = [5]
    X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq = get_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        threshold=threshold,
        use_smote=use_smote,
        sequence_length=sequence_length,
        as_tensor=False,
        flatten=True,
        idx_to_keep=idx_to_keep,
    )

    # normalize the data
    X_train_seq, X_val_seq, X_test_seq = normalize_data(
        X_train_seq, X_val_seq, X_test_seq
    )

    # Initialize visualizer
    visualizer = ModelVisualizer(save_dir="visualizations/random_forest")

    # Train model
    model = train_random_forest(
        X_train_seq,
        y_train_seq,
        X_val_seq,
        y_val_seq,
        optimize_hyperparams=optimize,
    )

    # Evaluate model on validation set
    val_metrics = evaluate_model(model, X_val_seq, y_val_seq, "Validation")

    # Evaluate model on test set
    test_metrics = evaluate_model(model, X_test_seq, y_test_seq, "Test")

    positive_to_negative_data = float(y_test_seq.sum() / len(y_test_seq))
    positive_to_negative_predicted = float(
        test_metrics["predictions"].sum() / len(test_metrics["predictions"])
    )

    # Add run to visualizer with metadata
    visualizer.add_evaluation(
        run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        eval_metrics=test_metrics,
        metadata={
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
            "min_samples_leaf": model.min_samples_leaf,
            "class_weight": model.class_weight,
            "optimize_hyperparams": optimize,
            "positive_to_negative_data": positive_to_negative_data,
            "positive_to_negative_predicted": positive_to_negative_predicted,
        },
    )

    # Generate visualizations
    visualizer.plot_confusion_matrices()
    visualizer.plot_roc_curves()
    visualizer.plot_precision_recall_curves()
    visualizer.save_run_metrics()

    # Save model
    save_model(model, model_output)


if __name__ == "__main__":
    args = parse_args()

    main(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        threshold=args.threshold,
        use_smote=args.use_smote,
        optimize=args.optimize,
    )
