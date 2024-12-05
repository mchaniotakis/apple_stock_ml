import torch
import torch.nn as nn
import torch.optim as optim
from torch import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
import argparse
from pathlib import Path
from datetime import datetime
from apple_stock_ml.src.data_preprocessing import get_data
from apple_stock_ml.src.temporal_cnn import (
    get_balanced_sampler,
    normalize_data,
    add_noise_to_sequence,
    mask_sequence_augmentation,
    EarlyStopping,
    compute_class_weight,
)
from apple_stock_ml import DEVICE
from apple_stock_ml.src.utils.logger import setup_logger
from apple_stock_ml.src.utils.visualizer import ModelVisualizer
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import math
import os
from tqdm import tqdm

# Set up logger
logger = setup_logger("train_timesnet", "logs/train_timesnet.log")


def parse_args():
    parser = argparse.ArgumentParser(description="Train TimesNet model")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--sequence-length", type=int, default=100
    )  # needs to be more for timesNet
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--model-output", type=str, default="models/timesnet.pt")
    parser.add_argument("--use-smote", action="store_true")
    parser.add_argument("--start-date", type=str, default="2015-01-01")
    parser.add_argument("--end-date", type=str, default="2024-01-31")
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--augs", action="store_true")

    return parser.parse_args()


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_X, batch_y, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X, None)
            probabilities = F.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            # Move everything to CPU and convert to numpy
            all_probs.extend(probabilities.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    # Convert lists to numpy arrays
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds)

    return {
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
        "predictions": all_preds,
        "true_values": all_labels,
        "probabilities": all_probs,
    }


def collate_fn(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [
        X.shape[0] for X in features
    ]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)

    X = torch.zeros(
        batch_size, max_len, features[0].shape[-1]
    )  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(
        torch.tensor(lengths, dtype=torch.int16), max_len=max_len
    )  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = (
        max_len or lengths.max_val()
    )  # trick works because of overloading of 'or' operator for non-boolean types
    return (
        torch.arange(0, max_len, device=lengths.device)
        .type_as(lengths)
        .repeat(batch_size, 1)
        .lt(lengths.unsqueeze(1))
    )


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(
        self,
        norm_type="standardization",
        mean=None,
        std=None,
        min_val=None,
        max_val=None,
    ):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (
                self.max_val - self.min_val + np.finfo(float).eps
            )

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform("mean")) / grouped.transform("std")

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform("min")
            return (df - min_vals) / (
                grouped.transform("max") - min_vals + np.finfo(float).eps
            )

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method="linear", limit_direction="both")
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            )
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=[1, 2 * i + 3],
                    padding=[0, i + 1],
                )
            )
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=[2 * i + 3, 1],
                    padding=[i + 1, 0],
                )
            )
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels // 2 * 2 + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


def FFT_for_Period(x, k=2):
    # Perform FFT to find dominant periods
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(
        self,
        sequence_length,
        input_dim,
        prediction_length=0,
        hidden_dim=256,
        num_kernels=6,
        top_k=3,
    ):
        super(TimesBlock, self).__init__()
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.top_k = top_k  # Number of top periods to consider

        # Simplified inception blocks
        self.conv = nn.Sequential(
            Inception_Block_V1(input_dim, hidden_dim, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(hidden_dim, input_dim, num_kernels=num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.top_k)

        res = []
        total_length = self.sequence_length + self.prediction_length
        for i in range(self.top_k):
            period = period_list[i]
            # Padding if necessary
            if total_length % period != 0:
                length = ((total_length // period) + 1) * period
                padding = torch.zeros(B, length - total_length, N).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = total_length
                out = x
            # Reshape
            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # 2D convolution
            out = self.conv(out)
            # Reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :total_length, :])
        res = torch.stack(res, dim=-1)
        # Adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = (
            period_weight.unsqueeze(1).unsqueeze(1).repeat(1, total_length, N, 1)
        )
        res = torch.sum(res * period_weight, dim=-1)
        # Residual connection
        res = res + x
        return res


def FFT_for_Period(x, k=3):
    B, T, N = x.size()
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = torch.abs(xf).mean(dim=0).mean(dim=-1)
    frequency_list[0] = 0  # Ignore the zero frequency component
    _, top_k_indices = torch.topk(frequency_list, k)
    top_k_indices = top_k_indices.detach().cpu().numpy()
    periods = T // top_k_indices
    period_weights = torch.abs(xf).mean(dim=-1)[:, top_k_indices]
    return periods, period_weights


class TokenEmbedding(nn.Module):
    def __init__(self, channels_in, dimension_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(
            in_channels=channels_in,
            out_channels=dimension_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        # x is B SL C
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, dimension_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dimension_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, dimension_model, 2).float()
            * -(math.log(10000.0) / dimension_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class FixedEmbedding(nn.Module):
    def __init__(self, channels_in, dimension_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(channels_in, dimension_model).float()
        w.require_grad = False

        position = torch.arange(0, channels_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, dimension_model, 2).float()
            * -(math.log(10000.0) / dimension_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(channels_in, dimension_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, dimension_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, dimension_model)
        self.hour_embed = Embed(hour_size, dimension_model)
        self.weekday_embed = Embed(weekday_size, dimension_model)
        self.day_embed = Embed(day_size, dimension_model)
        self.month_embed = Embed(month_size, dimension_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, dimension_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, dimension_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(
        self, input_channels, dimension_model, embed_type="fixed", freq="h", dropout=0.1
    ):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(
            channels_in=input_channels, dimension_model=dimension_model
        )
        self.position_embedding = PositionalEmbedding(dimension_model=dimension_model)
        self.temporal_embedding = (
            TemporalEmbedding(
                dimension_model=dimension_model, embed_type=embed_type, freq=freq
            )
            if embed_type != "timeF"
            else TimeFeatureEmbedding(
                dimension_model=dimension_model, embed_type=embed_type, freq=freq
            )
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (
                self.value_embedding(x)
                + self.temporal_embedding(x_mark)
                + self.position_embedding(x)
            )
        return self.dropout(x)


class TimesNet(nn.Module):
    def __init__(
        self,
        input_channels,
        sequence_length,
        num_classes=2,
        num_layers=2,
        dimension_model=64,
        dropout=0.1,
        freq="d",
        embed_type="timeF",
        prediction_length=0,
        hidden_dim=256,
        num_kernels=6,
        top_k=3,
    ):
        super(TimesNet, self).__init__()
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.num_layers = num_layers
        self.dimension_model = dimension_model

        # Input embedding
        self.enc_embedding = DataEmbedding(
            input_channels=input_channels,
            dimension_model=dimension_model,
            embed_type=embed_type,
            freq=freq,
            dropout=dropout,
        )

        # TimesBlocks
        self.model_layers = nn.ModuleList(
            [
                TimesBlock(
                    sequence_length=self.sequence_length,
                    prediction_length=self.prediction_length,
                    input_dim=dimension_model,
                    hidden_dim=hidden_dim,
                    num_kernels=num_kernels,
                    top_k=top_k,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(dimension_model)

        self.activation = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(dimension_model * self.sequence_length, num_classes)

    def forward(self, x_enc, x_mark_enc=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)  # Shape: [B, T, C]

        # TimesNet layers
        for layer in self.model_layers:
            enc_out = self.layer_norm(layer(enc_out))

        # Output
        output = self.activation(enc_out)
        output = self.dropout(output)
        # Reshape for classification
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output


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
    for _, labels, padding_mask in train_loader:
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
        for batch_X, batch_y, padding_mask in tqdm(
            train_loader, desc=f"Training epoch {epoch+1}"
        ):
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            padding_mask = padding_mask.to(DEVICE)
            if use_augmentations:
                batch_X = add_noise_to_sequence(batch_X)
                batch_X = mask_sequence_augmentation(batch_X)
            optimizer.zero_grad()
            outputs = model(batch_X, None)
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
            for batch_X, batch_y, padding_mask in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                padding_mask = padding_mask.to(DEVICE)
                outputs = model(batch_X, None)
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


def invert_channel_dimension(X):
    return X.permute(0, 2, 1)


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

    X_train = invert_channel_dimension(X_train)
    X_val = invert_channel_dimension(X_val)
    X_test = invert_channel_dimension(X_test)

    # Get balanced sampler
    sampler = get_balanced_sampler(X_train, y_train)

    # Create data loader with the sampler
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=lambda x: collate_fn(x, max_len=args.sequence_length),
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val),
        batch_size=args.batch_size,
        collate_fn=lambda x: collate_fn(x, max_len=args.sequence_length),
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test),
        batch_size=args.batch_size,
        collate_fn=lambda x: collate_fn(x, max_len=args.sequence_length),
    )

    # Initialize models
    model = TimesNet(
        input_channels=X_train.shape[2],  # padded to sequence length
        sequence_length=args.sequence_length,
        dropout=0.1,
        freq="d",
        embed_type="timeF",
        num_classes=2,
        dimension_model=64,
        num_layers=3,
        top_k=1,
        prediction_length=0,
    )

    # Initialize visualizer
    visualizer = ModelVisualizer(save_dir="visualizations/timesnet")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_metrics = evaluate_model(model, test_loader, device)

    # Add run to visualizer
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

    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    # Save model
    torch.save(model.state_dict(), args.model_output)


if __name__ == "__main__":
    main()
