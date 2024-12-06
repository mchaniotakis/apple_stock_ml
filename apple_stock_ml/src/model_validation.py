from typing import Dict, List, Optional
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from pathlib import Path
import argparse
from apple_stock_ml.src.data_preprocessing import get_data
from apple_stock_ml.src.random_forest import evaluate_model as evaluate_rf
from apple_stock_ml.src.random_forest import load_model as load_rf_model
from apple_stock_ml.src.temporal_cnn import evaluate_model as evaluate_cnn
from apple_stock_ml.src.temporal_cnn import load_model as load_tcnn_model
from apple_stock_ml.src.improvement import evaluate_model as evaluate_improved
from apple_stock_ml.src.improvement import load_model as load_improved_model
from apple_stock_ml.src.improvement import collate_fn, invert_channel_dimension
from apple_stock_ml.src.utils.visualizer import ModelVisualizer
import ast
import torch

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, save_dir: str = "evaluation_results"):
        """Initialize the model evaluator.

        Args:
            save_dir: Directory to save evaluation results
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = ModelVisualizer(save_dir=self.save_dir)

    def evaluate_all_models(
        self,
        ticker: str = "AAPL",
        start_date: str = "2015-01-01",
        end_date: str = "2024-01-31",
        threshold: float = 2.0,
        pca: bool = False,
        sequence_length: int = 10,
        rf_idx_to_keep: List[int] = None,
        # test_loader: torch.utils.data.DataLoader = None,
    ) -> Dict:
        """Evaluate all models and compare their performance.

        Returns:
            Dict containing evaluation metrics for all models
        """

        logger.info("Fetching data for random forrest")

        X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq = (
            get_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                threshold=threshold,
                use_smote=False,
                sequence_length=sequence_length,
                as_tensor=False,
                flatten=True,
                idx_to_keep=rf_idx_to_keep,
            )
        )

        # load latest model
        model = load_rf_model(f"models/random_forest.pkl")

        # Evaluate Random Forest
        rf_metrics = evaluate_rf(
            model=model,
            X=X_test_seq,
            y=y_test_seq,
            set_name="Test",
        )
        self.visualizer.add_evaluation(
            "Random Forest", rf_metrics, metadata={"model_type": "random_forest"}
        )

        X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq = (
            get_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                threshold=threshold,
                use_smote=False,
                sequence_length=sequence_length,
                as_tensor=True,
            )
        )
        # create dataloaders for temporal CNN + improved model
        test_data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test_seq, y_test_seq),
            batch_size=4,
        )
        # Evaluate Temporal CNN
        model = load_tcnn_model("models/temporal_cnn.pt")
        cnn_metrics = evaluate_cnn(
            model=model,
            test_loader=test_data_loader,
            pca=pca,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            threshold=threshold,
            sequence_length=sequence_length,
        )  # You'll need to pass the appropriate test_loader
        self.visualizer.add_evaluation(
            "Temporal CNN", cnn_metrics, metadata={"model_type": "temporal_cnn"}
        )

        model = load_improved_model("models/improved.pt")
        # create dataloaders for temporal CNN + improved model
        X_test_seq = invert_channel_dimension(X_test_seq)
        test_data_loader_improved = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test_seq, y_test_seq),
            batch_size=16,
            collate_fn=lambda x: collate_fn(x, max_len=sequence_length),
        )
        # Evaluate Improved Model
        improved_metrics = evaluate_improved(
            model=model,
            test_loader=test_data_loader_improved,
            pca=pca,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            threshold=threshold,
            sequence_length=sequence_length,
        )  # You'll need to pass appropriate parameters
        self.visualizer.add_evaluation(
            "Improved Model", improved_metrics, metadata={"model_type": "improved"}
        )

        # Generate comparison visualizations
        self._generate_comparison_plots()

        # Save detailed metrics
        self.visualizer.save_run_metrics()

        return {
            "random_forest": rf_metrics,
            "temporal_cnn": cnn_metrics,
            "improved": improved_metrics,
        }

    def _generate_comparison_plots(self):
        """Generate comparison plots for all models."""
        # Plot ROC curves
        self.visualizer.plot_roc_curves()

        # Plot Precision-Recall curves
        self.visualizer.plot_precision_recall_curves()

        # Print detailed classification reports
        for model_name in ["Random Forest", "Temporal CNN", "Improved Model"]:
            print(f"\n=== {model_name} Classification Report ===")
            self.visualizer.print_classification_report(model_name)

    def generate_summary_report(self):
        """Generate a summary report comparing all models."""
        report = {"timestamp": datetime.now().isoformat(), "models": {}}

        for run_name, run_data in self.visualizer.runs.items():
            metrics = run_data.get("classification_report", {})
            report["models"][run_name] = {
                "accuracy": metrics.get("accuracy", 0),
                "macro_avg_f1": metrics.get("macro avg", {}).get("f1-score", 0),
                "weighted_avg_f1": metrics.get("weighted avg", {}).get("f1-score", 0),
            }

        # Save summary report
        with open(self.save_dir / "summary_report.json", "w") as f:
            json.dump(report, f, indent=2)

        return report


def main(
    ticker: str = "AAPL",
    start_date: str = "2015-01-01",
    end_date: str = "2024-01-31",
    threshold: float = 2.0,
    rf_idx_to_keep: List[int] = None,
    save_dir: str = "evaluation_results",
):
    # Initialize evaluator
    evaluator = ModelEvaluator(save_dir=save_dir)

    # Run evaluation for all models
    results = evaluator.evaluate_all_models(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        threshold=threshold,
        rf_idx_to_keep=rf_idx_to_keep,
    )

    # Generate summary report
    summary = evaluator.generate_summary_report()
    print(summary)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate and compare different stock prediction models"
    )
    parser.add_argument(
        "--ticker", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2015-01-01",
        help="Start date for evaluation (default: 2015-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-01-31",
        help="End date for evaluation (default: 2024-01-31)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Threshold for classification (default: 2.0)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results (default: evaluation_results)",
    )
    parser.add_argument(
        "--rf_idx_to_keep",
        type=str,
        default=None,
        help="Indices to keep for random forest (default: -1)",
    )

    args = parser.parse_args()
    if args.rf_idx_to_keep:
        rf_idx_to_keep = ast.literal_eval(args.rf_idx_to_keep)
    else:
        rf_idx_to_keep = None

    main(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        threshold=args.threshold,
        save_dir=args.save_dir,
        rf_idx_to_keep=rf_idx_to_keep,
    )
