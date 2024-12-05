"""Visualization utilities for model training and evaluation."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
from sklearn.metrics import roc_curve, auc, precision_recall_curve


class ModelVisualizer:
    """Class for visualizing and comparing model runs."""

    def __init__(self, save_dir: str = "visualizations"):
        """Initialize the visualizer.

        Args:
            save_dir: Directory to save visualization outputs
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.runs: Dict[str, Dict] = {}

    def add_run(
        self,
        run_name: str,
        train_losses: List[float],
        val_losses: List[float],
        predictions: np.ndarray,
        true_values: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        metadata: Optional[Dict] = None,
    ):
        """Add a model run to the visualizer.

        Args:
            run_name: Unique identifier for the run
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
            predictions: Model predictions
            true_values: True labels
            dates: Optional dates for time series plotting
            metadata: Optional metadata about the run (hyperparameters etc.)
        """
        self.runs[run_name] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "predictions": predictions,
            "true_values": true_values,
            "dates": dates,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

    def add_evaluation(
        self,
        run_name: str,
        eval_metrics: Dict,
        train_losses: Optional[List[float]] = None,
        val_losses: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
    ):
        """Add evaluation results to the visualizer.

        Args:
            run_name: Unique identifier for the run
            eval_metrics: Dictionary containing evaluation metrics
            train_losses: Optional list of training losses
            val_losses: Optional list of validation losses
            metadata: Optional metadata about the run
        """
        self.runs[run_name] = {
            "train_losses": train_losses or [],
            "val_losses": val_losses or [],
            "predictions": eval_metrics["predictions"],
            "true_values": eval_metrics["true_values"],
            "probabilities": eval_metrics.get("probabilities", None),
            "test_loss": eval_metrics.get("test_loss", None),
            "confusion_matrix": eval_metrics.get("confusion_matrix", None),
            "classification_report": eval_metrics.get("classification_report", None),
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

    def plot_pca_variance(
        self, variance_ratio, cumulative_variance, n_optimal, save: bool = True
    ):
        """Plot PCA explained variance ratio.

        Args:
            variance_ratio: Array of explained variance ratios for each component
            cumulative_variance: Array of cumulative explained variance ratios
            n_optimal: Optimal number of components based on variance threshold
            save: Whether to save the plot to disk
        """
        plt.figure(figsize=(10, 6))

        # Individual variance plot
        plt.bar(
            range(1, len(variance_ratio) + 1),
            variance_ratio,
            alpha=0.5,
            label="Individual explained variance",
        )

        # Cumulative variance plot
        plt.step(
            range(1, len(cumulative_variance) + 1),
            cumulative_variance,
            where="mid",
            label="Cumulative explained variance",
        )

        # Mark optimal components
        plt.axvline(
            x=n_optimal,
            color="r",
            linestyle="--",
            label=f"Optimal components: {n_optimal}",
        )

        plt.xlabel("Number of Components")
        plt.ylabel("Explained Variance Ratio")
        plt.title("PCA Explained Variance Analysis")
        plt.legend()
        plt.grid(True)

        if save:
            plt.savefig(self.save_dir / "pca_variance.png")
            plt.close()
        else:
            plt.show()

    def plot_extreme_events(
        self, ticker: str, df: pd.DataFrame, threshold: float, save: bool = True
    ):
        """Plot daily returns and highlight extreme events.

        Args:
            df: DataFrame containing Daily_Return and Extreme_Event columns
            threshold: Threshold value used for extreme event detection
            save: Whether to save the plot to disk
        """
        plt.figure(figsize=(55, 8))

        # Plot daily returns
        plt.plot(df.index, df["Daily_Return"], "b-", alpha=0.3, label="Daily Returns")

        # Highlight extreme events
        extreme_dates = df[df["Extreme_Event"].shift(1) == 1].index
        extreme_returns = df.loc[extreme_dates, "Daily_Return"]
        plt.scatter(
            extreme_dates,
            extreme_returns,
            color="red",
            alpha=0.6,
            label="Extreme Events (Next Day)",
            zorder=5,
        )

        # Add threshold lines
        plt.axhline(y=threshold, color="r", linestyle="--", alpha=0.3)
        plt.axhline(y=-threshold, color="r", linestyle="--", alpha=0.3)

        plt.title(f"Daily Returns and Extreme Events (Threshold: ±{threshold}%)")
        plt.xlabel("Date")
        plt.ylabel("Daily Return (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save:
            plt.savefig(self.save_dir / f"{ticker}_extreme_events_{threshold}.png")
            plt.close()
        else:
            plt.show()

    def plot_learning_curves(
        self, run_names: Optional[List[str]] = None, save: bool = True
    ):
        """Plot learning curves for specified runs.

        Args:
            run_names: List of run names to plot. If None, plot all runs.
            save: Whether to save the plot to disk
        """
        plt.figure(figsize=(12, 6))
        run_names = run_names or list(self.runs.keys())

        for run_name in run_names:
            run = self.runs[run_name]
            epochs = range(1, len(run["train_losses"]) + 1)
            plt.plot(epochs, run["train_losses"], "--", label=f"{run_name} (train)")
            plt.plot(epochs, run["val_losses"], "-", label=f"{run_name} (val)")

        plt.title("Learning Curves Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        if save:
            plt.savefig(self.save_dir / "learning_curves.png")
            plt.close()
        else:
            plt.show()

    def plot_confusion_matrices(
        self, run_names: Optional[List[str]] = None, save: bool = True
    ):
        """Plot confusion matrices for specified runs.

        Args:
            run_names: List of run names to plot. If None, plot all runs.
            save: Whether to save the plot to disk
        """
        run_names = run_names or list(self.runs.keys())
        n_runs = len(run_names)
        fig, axes = plt.subplots(1, n_runs, figsize=(6 * n_runs, 5))
        if n_runs == 1:
            axes = [axes]

        for ax, run_name in zip(axes, run_names):
            run = self.runs[run_name]
            cm = pd.crosstab(run["true_values"], run["predictions"])
            sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
            ax.set_title(f"Confusion Matrix - {run_name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

        plt.tight_layout()
        if save:
            plt.savefig(self.save_dir / "confusion_matrices.png")
            plt.close()
        else:
            plt.show()

    def plot_prediction_timeline(
        self,
        run_names: Optional[List[str]] = None,
        window: Optional[Tuple[int, int]] = None,
        save: bool = True,
    ):
        """Plot predictions over time for specified runs.

        Args:
            run_names: List of run names to plot. If None, plot all runs.
            window: Optional tuple of (start_idx, end_idx) to zoom into specific region
            save: Whether to save the plot to disk
        """
        plt.figure(figsize=(15, 6))
        run_names = run_names or list(self.runs.keys())

        for run_name in run_names:
            run = self.runs[run_name]
            if run["dates"] is not None:
                x = run["dates"]
            else:
                x = range(len(run["predictions"]))

            if window:
                start, end = window
                x = x[start:end]
                preds = run["predictions"][start:end]
                true = run["true_values"][start:end]
            else:
                preds = run["predictions"]
                true = run["true_values"]

            plt.plot(x, preds, "o", label=f"{run_name} (pred)", alpha=0.6)
            plt.plot(x, true, "-", label=f"{run_name} (true)", alpha=0.6)

        plt.title("Predictions Timeline")
        plt.xlabel("Date" if isinstance(x, pd.DatetimeIndex) else "Time Step")
        plt.ylabel("Extreme Event")
        plt.legend()
        plt.grid(True)

        if save:
            plt.savefig(self.save_dir / "prediction_timeline.png")
            plt.close()
        else:
            plt.show()

    def save_run_metrics(self):
        """Save run metrics and metadata to disk."""
        metrics = {}
        for run_name, run in self.runs.items():
            metric_dict = {
                "metadata": run["metadata"],
                "timestamp": run["timestamp"],
                "accuracy": (run["predictions"] == run["true_values"]).mean(),
            }

            # Add loss metrics if they exist
            if run["train_losses"]:
                metric_dict["final_train_loss"] = run["train_losses"][-1]
            if run["val_losses"]:
                metric_dict["final_val_loss"] = run["val_losses"][-1]
            if run.get("test_loss") is not None:
                metric_dict["test_loss"] = run["test_loss"]

            metrics[run_name] = metric_dict

        with open(self.save_dir / "run_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    def plot_roc_curves(self, run_names: Optional[List[str]] = None, save: bool = True):
        """Plot ROC curves for specified runs.

        Args:
            run_names: List of run names to plot. If None, plot all runs.
            save: Whether to save the plot to disk
        """
        plt.figure(figsize=(10, 6))
        run_names = run_names or list(self.runs.keys())

        for run_name in run_names:
            run = self.runs[run_name]
            # Get probabilities for positive class
            y_score = run["probabilities"][:, 1]
            fpr, tpr, _ = roc_curve(run["true_values"], y_score)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"{run_name} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves Comparison")
        plt.legend(loc="lower right")
        plt.grid(True)

        if save:
            plt.savefig(self.save_dir / "roc_curves.png")
            plt.close()
        else:
            plt.show()

    def plot_precision_recall_curves(
        self, run_names: Optional[List[str]] = None, save: bool = True
    ):
        """Plot Precision-Recall curves for specified runs.

        Args:
            run_names: List of run names to plot. If None, plot all runs.
            save: Whether to save the plot to disk
        """
        plt.figure(figsize=(10, 6))
        run_names = run_names or list(self.runs.keys())

        for run_name in run_names:
            run = self.runs[run_name]
            y_score = run["probabilities"][:, 1]
            precision, recall, _ = precision_recall_curve(run["true_values"], y_score)
            pr_auc = auc(recall, precision)

            plt.plot(recall, precision, label=f"{run_name} (AUC = {pr_auc:.2f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves Comparison")
        plt.legend(loc="lower left")
        plt.grid(True)

        if save:
            plt.savefig(self.save_dir / "precision_recall_curves.png")
            plt.close()
        else:
            plt.show()

    def print_classification_report(self, run_name: str):
        """Print detailed classification report for a specific run.

        Args:
            run_name: Name of the run to report
        """
        if run_name not in self.runs:
            raise ValueError(f"Run '{run_name}' not found")

        run = self.runs[run_name]
        report = run["classification_report"]

        print(f"\nClassification Report for {run_name}")
        print("-" * 60)

        # Print metrics for each class
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"\nClass: {label}")
                print(f"Precision: {metrics['precision']:.3f}")
                print(f"Recall: {metrics['recall']:.3f}")
                print(f"F1-score: {metrics['f1-score']:.3f}")
                print(f"Support: {metrics['support']}")

        # Print average metrics
        print("\nOverall Metrics:")
        print(f"Accuracy: {report['accuracy']:.3f}")
        print(f"Macro avg F1: {report['macro avg']['f1-score']:.3f}")
        print(f"Weighted avg F1: {report['weighted avg']['f1-score']:.3f}")
