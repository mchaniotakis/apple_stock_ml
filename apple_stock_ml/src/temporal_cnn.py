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
from torch.nn import functional as F
from apple_stock_ml.src.utils.visualizer import ModelVisualizer
from tqdm import tqdm
from apple_stock_ml import DEVICE, set_seeds

# Set up logger
logger = setup_logger("temporal_cnn", "logs/temporal_cnn.log")


def normalize_data(data, except_channels=[-1]):
    """Normalize each channel independently using training set statistics, except for the last channel"""
    # Get all channels except the last one
    channels_to_normalize = data.shape[1] - 1

    # Initialize normalized tensors
    data_norm = data.clone()

    # Normalize each channel independently
    for channel in range(channels_to_normalize):
        if channel in except_channels:
            continue
        # Calculate statistics for current channel
        mean = data[:, channel : channel + 1].mean(dim=0, keepdim=True)
        std = data[:, channel : channel + 1].std(dim=0, keepdim=True)

        # Normalize current channel
        data_norm[:, channel : channel + 1] = (
            data[:, channel : channel + 1] - mean
        ) / (std + 1e-8)

    return data


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
            input_channels = input_channels if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TCNNBlock(
                    input_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dropout_rate=dropout_rate,
                    dilation=dilation,
                )
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 2)
        self.dropout_final = nn.Dropout(dropout_rate)
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
        x = self.dropout_final(x)
        return x

    def predict(self, x):
        return self.softmax(self.forward(x))


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=7, min_delta=0.001):
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


def save_model(model, filepath):
    """Save the trained model to disk."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Create dictionary with model parameters and state
    model_data = {
        "state_dict": model.state_dict(),
        "input_channels": model.network[0].conv1.in_channels,
        "num_channels": [layer.conv1.out_channels for layer in model.network],
        "kernel_size": model.network[0].conv1.kernel_size[0],
        "dropout_rate": model.network[0].dropout1.p,
    }
    torch.save(model_data, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath):
    """Load a saved model from disk.

    Args:
        filepath (str): Path to the saved model data

    Returns:
        TemporalCNN: Loaded model
    """
    model_data = torch.load(filepath)

    # Initialize model with saved parameters
    model = TemporalCNN(
        input_channels=model_data["input_channels"],
        num_channels=model_data["num_channels"],
        kernel_size=model_data["kernel_size"],
        dropout_rate=model_data["dropout_rate"],
    )
    model.load_state_dict(model_data["state_dict"])
    return model


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


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def main(
    ticker: str,
    start_date: str = "2015-01-01",
    end_date: str = "2024-01-31",
    threshold: float = 2.0,
    sequence_length: int = 10,
    use_smote: bool = False,
    pca: bool = False,
    batch_size: int = 2,
    model_output: str = "models/temporal_cnn.pt",
):
    set_seeds()
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = get_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        threshold=threshold,
        use_smote=use_smote,
        sequence_length=sequence_length,
        as_tensor=True,
        apply_pca=pca,
        # idx_to_keep=[-1],
    )

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
        batch_size=batch_size,
        sampler=sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val), batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test), batch_size=batch_size
    )

    # Initialize model

    model = TemporalCNN(
        input_channels=X_train.shape[1],
        num_channels=[64, 128],
        kernel_size=3,
        dropout_rate=0.1,
    )
    model.init_weights()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # Define a lambda function for the warm-up phase
    def warmup_lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        return 1.0

    # Set the number of warm-up epochs
    warmup_epochs = 5

    # Create a LambdaLR scheduler for the warm-up
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warmup_lr_lambda
    )

    # Create a ReduceLROnPlateau scheduler for after the warm-up
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6,
        threshold=0.01,
        threshold_mode="rel",
    )
    # Initialize visualizer
    visualizer = ModelVisualizer(save_dir="visualizations/temporal_cnn")

    # Train model
    model, train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        warmup_epochs,
        warmup_scheduler,
        plateau_scheduler,
        use_augmentations=args.augs,
        num_epochs=args.epochs,
        patience=args.patience,
    )

    # Evaluate model
    eval_metrics = evaluate_model(model, test_loader)

    # Calculate positive and negative prediction rates
    positive_to_negative_data = float((y_test.sum() / len(y_test)).cpu().numpy())
    positive_to_negative_predicted = (eval_metrics["predictions"] == 1).sum() / len(
        eval_metrics["predictions"]
    )
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
            "positive_to_negative_data": positive_to_negative_data,
            "positive_to_negative_predicted": positive_to_negative_predicted,
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


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    warmup_epochs,
    warmup_scheduler,
    plateau_scheduler,
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
    # criterion = FocalLoss(alpha=weights)
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
            batch_X = normalize_data(batch_X)
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            if use_augmentations:
                batch_X = add_noise_to_sequence(batch_X)
                batch_X = mask_sequence_augmentation(batch_X)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # # Add gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = normalize_data(batch_X)
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                if best_model is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if epoch > warmup_epochs:
            plateau_scheduler.step(
                avg_val_loss
            )  # Pass validation loss to plateau scheduler
        else:
            warmup_scheduler.step()

        # Log progress
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"LR: {optimizer.param_groups[0]['lr']}")
        logger.info(f"Training Loss: {avg_train_loss:.4f}")
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")

        # Early stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    return best_model, train_losses, val_losses


def evaluate_model(
    model: nn.Module = None,
    test_loader: torch.utils.data.DataLoader = None,
    pca: bool = False,
    ticker: str = "AAPL",
    start_date: str = "2015-01-01",
    end_date: str = "2024-01-31",
    threshold: float = 2.0,
    sequence_length: int = 10,
) -> dict:
    """Evaluate the model and print performance metrics.

    Returns:
        dict: Dictionary containing evaluation metrics and predictions
    """

    if isinstance(model, type(None)):
        # load from expected location
        model_path = "models/temporal_cnn.pt"
        model = torch.load(model_path)
    if isinstance(test_loader, type(None)):
        # get test data
        _, _, X_test, _, _, y_test = get_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            threshold=threshold,
            sequence_length=sequence_length,
            as_tensor=True,
            apply_pca=pca,
        )

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test, y_test), batch_size=batch_size
        )
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []  # Store probability predictions
    test_loss = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            # normalize data
            batch_X = normalize_data(batch_X)
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model.predict(batch_X)
            predicted = 1 - outputs.argmax(dim=1)
            all_probs.extend(outputs.cpu().numpy())
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


def evaluate(
    model_path,
    ticker="AAPL",
    start_date="2015-01-01",
    end_date="2024-01-31",
    threshold=2.0,
    sequence_length=10,
    batch_size=32,
    apply_pca=False,
):
    """Load a saved model and evaluate it on the test set.

    Args:
        model_path (str): Path to the saved model
        ticker (str): Stock ticker symbol
        start_date (str): Start date for data
        end_date (str): End date for data
        threshold (float): Threshold for extreme events
        sequence_length (int): Length of input sequences
        batch_size (int): Batch size for evaluation
        apply_pca (bool): Whether to apply PCA to sequences

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Get test data
    _, _, X_test, _, _, y_test = get_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        threshold=threshold,
        sequence_length=sequence_length,
        as_tensor=True,
        apply_pca=apply_pca,
    )

    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test), batch_size=batch_size
    )

    # Load model
    model = TemporalCNN(
        input_channels=X_test.shape[1],
        num_channels=[16, 32],
        kernel_size=3,
        dropout_rate=0.2,
    )
    model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)

    # Evaluate
    metrics = evaluate_model(model, test_loader)

    return metrics


if __name__ == "__main__":
    args = parse_args()
    main(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        threshold=args.threshold,
        sequence_length=args.sequence_length,
        use_smote=args.use_smote,
        pca=args.pca,
        batch_size=args.batch_size,
        model_output=args.model_output,
    )
