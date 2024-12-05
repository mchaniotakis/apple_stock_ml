import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
import argparse
from pathlib import Path
from datetime import datetime
from apple_stock_ml.src.data_preprocessing import (
    load_stock_data,
    prepare_sp500_features,
    split_time_series_data,
)
from apple_stock_ml.src.temporal_cnn import EarlyStopping, create_sequences
from apple_stock_ml.src.utils.logger import setup_logger
from apple_stock_ml.src.utils.visualizer import ModelVisualizer
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import math

# Set up logger
logger = setup_logger("train_timesnet", "logs/train_timesnet.log")


def parse_args():
    parser = argparse.ArgumentParser(description="Train TimesNet model")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--sequence-length", type=int, default=100
    )  # needs to be more for timesNet
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--model-output", type=str, default="models/timesnet.pt")
    parser.add_argument("--use-smote", action="store_true")
    parser.add_argument("--start-date", type=str, default="2015-01-01")
    parser.add_argument("--end-date", type=str, default="2024-01-31")
    parser.add_argument("--threshold", type=float, default=2.0)
    return parser.parse_args()


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X, None, None, None)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(predictions)
            all_labels.extend(batch_y.numpy())

    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds)

    return {
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
        "predictions": all_preds,
        "true_labels": all_labels,
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=args.patience)

    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()

            # Use autocast for mixed precision training
            with autocast():
                outputs = model(batch_X, None, None, None)
                loss = criterion(outputs, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X, None, None, None)
                val_loss += criterion(outputs, batch_y).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"Training Loss: {avg_train_loss:.4f}")
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    return model, train_losses, val_losses


def main():
    args = parse_args()

    # Load and preprocess data
    data = load_stock_data(
        start_date=args.start_date,
        end_date=args.end_date,
        ticker="SP500",
        cache_path=Path("data/cache") / f"SP500_{args.start_date}_{args.end_date}.pkl",
    )

    # Prepare features
    X, y = prepare_sp500_features(data, threshold=args.threshold)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_time_series_data(
        X, y, use_smote=args.use_smote
    )

    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, args.sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, args.sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, args.sequence_length)

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train_seq, y_train_seq),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_seq, y_val_seq), batch_size=args.batch_size
    )
    test_loader = DataLoader(
        TensorDataset(X_test_seq, y_test_seq), batch_size=args.batch_size
    )

    # Initialize model
    model = TimesNet(
        input_channels=X.shape[1],
        sequence_length=args.sequence_length,
        encoding_layers=2,
        dropout=0.1,
        freq="d",
        embed_type="timeF",
        num_classes=2,
        dimention_model=64,
        output_size=7,
        num_layers=2,
        top_k=3,
        label_length=48,
    )

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initialize visualizer
    visualizer = ModelVisualizer(save_dir="visualizations/timesnet")

    # Train model
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, args
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

    # Save model
    torch.save(model.state_dict(), args.model_output)


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
    def __init__(self, seq_len, in_channels, d_ff=256, num_kernels=6, top_k=2):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.k = top_k  # frequences to consider
        self.in_channels = in_channels
        self.d_ff = d_ff
        self.num_kernels = num_kernels

        # Simplified inception blocks
        self.conv = nn.Sequential(
            Inception_Block_V1(in_channels, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, in_channels, num_kernels=num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]
                ).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # reshape
            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : (self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TokenEmbedding(nn.Module):
    def __init__(self, channels_in, dimention_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=channels_in,
            out_channels=dimention_model,
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
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, dimention_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dimention_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, dimention_model, 2).float()
            * -(math.log(10000.0) / dimention_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

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
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(
        self, input_channels, d_model, embed_type="fixed", freq="h", dropout=0.1
    ):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=input_channels, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
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
        encoding_layers,
        dropout=0.1,
        freq="d",
        embed_type="timeF",
        num_classes=2,
        dimention_model=64,
        output_size=7,
        num_layers=2,
        top_k=3,  # T for imesBlock
        label_length=48,  # start token length
    ):
        super(TimesNet, self).__init__()
        self.encoding_layers = encoding_layers
        # Input embedding
        self.enc_embedding = DataEmbedding(
            input_channels,
            dimention_model,
            embed_type,
            freq,
            dropout,
        )

        self.layer = encoding_layers
        self.layer_norm = nn.LayerNorm(dimention_model)

        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(dimention_model * sequence_length, num_classes)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output


if __name__ == "__main__":
    main()
