"""Temporal CNN model for predicting extreme events in Apple stock prices."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from apple_stock_ml.src.utils.logger import setup_logger
from apple_stock_ml.src.data_preprocessing import (
    add_noise_to_sequence,
    mask_sequence_augmentation,
    get_data,
)
from apple_stock_ml.src.utils.visualizer import ModelVisualizer
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set up logger
logger = setup_logger("temporal_cnn", "logs/temporal_cnn.log")


def normalize_data(X_train, X_val, X_test):
    """Normalize the data using training set statistics"""
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True)

    X_train_norm = (X_train - mean) / (std + 1e-8)
    X_val_norm = (X_val - mean) / (std + 1e-8)
    X_test_norm = (X_test - mean) / (std + 1e-8)

    return X_train_norm, X_val_norm, X_test_norm


class TCNNBlock(nn.Module):
    """Temporal CNN block with residual connection and weight normalization."""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, dropout_rate=0.5, dilation=1
    ):
        super(TCNNBlock, self).__init__()

        self.dilation = dilation
        padding = (kernel_size - 1) * dilation // 2  # Causal padding

        # First dilated convolution
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second dilated convolution
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        # Residual connection if input/output dimensions differ
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # Store residual
        residual = x if self.downsample is None else self.downsample(x)

        # Apply convolutions
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Add residual and remove padding
        return (out + residual)[:, :, : -self.dilation * 2]


class TemporalCNN(nn.Module):
    """
    Temporal CNN model for sequence classification. Implmentation
    from the original paper https://arxiv.org/pdf/1803.01271
    """

    def __init__(self, input_channels, num_channels, kernel_size=3, dropout_rate=0.5):
        """
        Args:
            input_channels (int): Number of input features
            num_channels (list): Number of channels in each layer
            kernel_size (int): Kernel size for all convolutions
            dropout_rate (float): Dropout rate
        """
        super(TemporalCNN, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2**i  # Exponentially increasing dilation
            in_ch = input_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]

            layers.append(
                TCNNBlock(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    dropout_rate=dropout_rate,
                    dilation=dilation,
                )
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 2)
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.network(x)
        # Global average pooling
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return x  # self.softmax(x)


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    use_augmentations=True,
    num_epochs=100,
    patience=7,
):
    """Train the temporal CNN model."""
    model = model.to(DEVICE)

    # Calculate class weights
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(all_labels), y=all_labels
    )
    weights = torch.FloatTensor(class_weights).to(DEVICE)

    # Update criterion with weights
    criterion = nn.CrossEntropyLoss(weight=weights)

    early_stopping = EarlyStopping(patience=patience)

    train_losses = []
    val_losses = []

    best_model = None
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            if use_augmentations:
                batch_X = add_noise_to_sequence(batch_X)
                batch_X = mask_sequence_augmentation(batch_X)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                if best_model is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)

        # Log progress
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Training Loss: {avg_train_loss:.4f}")
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")

        # Early stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    return model, train_losses, val_losses


def evaluate_model(model, test_loader):
    """Evaluate the model and print performance metrics.

    Returns:
        dict: Dictionary containing evaluation metrics and predictions
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []  # Store probability predictions
    test_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Calculate classification report
    report = classification_report(all_labels, all_preds, output_dict=True)

    # Log results
    logger.info("\nTest Results:")
    logger.info(f"Average Test Loss: {test_loss/len(test_loader):.4f}")
    logger.info("\nConfusion Matrix:")
    logger.info(cm)
    logger.info("\nClassification Report:")
    logger.info(classification_report(all_labels, all_preds))

    # Return comprehensive metrics
    metrics = {
        "predictions": all_preds,
        "true_values": all_labels,
        "probabilities": all_probs,
        "test_loss": test_loss / len(test_loader),
        "confusion_matrix": cm,
        "classification_report": report,
    }

    return metrics


def save_model(model, filepath):
    """Save the trained model to disk."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    logger.info(f"Model saved to {filepath}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate Temporal CNN model"
    )
    parser.add_argument(
        "--sequence-length", type=int, default=10, help="Length of input sequences"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Patience for early stopping"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="models/temporal_cnn.pt",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--use-smote", action="store_true", help="Apply SMOTE for class balancing"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2015-01-01",
        help="Start date for data loading",
    )
    parser.add_argument(
        "--end-date", type=str, default="2024-01-31", help="End date for data loading"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="data/cache", help="Directory to cache data"
    )
    parser.add_argument(
        "--threshold", type=float, default=2.0, help="Threshold for extreme events"
    )
    parser.add_argument(
        "--ticker", type=str, default="AAPL", help="Stock ticker to use"
    )
    parser.add_argument(
        "--augs",
        action="store_true",
        help="Whether to apply augmentations (noise and masking)",
    )
    parser.add_argument("--pca", action="store_true", help="Apply PCA to sequences")
    return parser.parse_args()


def get_balanced_sampler(X, y):
    """Get a balanced sampler for the data.

    Args:
        X (torch.Tensor): Input features
        y (torch.Tensor): Target labels

    Returns:
        WeightedRandomSampler: Sampler that balances class distributions
    """
    y = np.array(y)
    # Calculate class weights
    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)

    # Create sample weights based on class weights
    sample_weights = [class_weights[t] for t in y]

    # Convert to tensor
    weights = torch.DoubleTensor(sample_weights)

    # Create and return the sampler
    return torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )


def main():
    args = parse_args()

    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = get_data(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        threshold=args.threshold,
        use_smote=args.use_smote,
        sequence_length=args.sequence_length,
        as_tensor=True,
        apply_pca=args.pca,
    )
    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)

    # print statistics about the data
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"y_val shape: {y_val.shape}")
    logger.info(f"y_test shape: {y_test.shape}")

    # Get balanced sampler
    sampler = get_balanced_sampler(X_train, y_train)

    # Create data loader with the sampler
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        sampler=sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val), batch_size=args.batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test), batch_size=args.batch_size
    )

    # Initialize model

    model = TemporalCNN(
        input_channels=X_train.shape[1],
        num_channels=[64, 128],
        kernel_size=3,
        dropout_rate=0.2,
    )
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # Spend 30% of time warming up
        div_factor=25,  # Initial lr = max_lr/25
        final_div_factor=1e4,  # Final lr = max_lr/10000
    )
    # Initialize visualizer
    visualizer = ModelVisualizer(save_dir="visualizations/temporal_cnn")

    # Train model
    model, train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        use_augmentations=args.augs,
        num_epochs=args.epochs,
        patience=args.patience,
    )

    # Evaluate model
    eval_metrics = evaluate_model(model, test_loader)

    # Add run to visualizer with metadata
    visualizer.add_evaluation(
        run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        eval_metrics=eval_metrics,
        train_losses=train_losses,
        val_losses=val_losses,
        metadata={
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "patience": args.patience,
        },
    )

    # Generate visualizations
    visualizer.plot_learning_curves()
    visualizer.plot_confusion_matrices()
    visualizer.plot_roc_curves()
    visualizer.plot_precision_recall_curves()
    visualizer.save_run_metrics()

    # Save model
    save_model(model, args.model_output)


if __name__ == "__main__":
    main()
