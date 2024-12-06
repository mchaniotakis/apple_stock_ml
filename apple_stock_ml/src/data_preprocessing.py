import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse
from pathlib import Path
import pickle
import os
from matplotlib import pyplot as plt
from apple_stock_ml.src.utils.logger import setup_logger
from apple_stock_ml.src.utils.visualizer import ModelVisualizer
from apple_stock_ml import set_seeds, SEED
from imblearn.over_sampling import SMOTE
from collections import Counter
from tqdm import tqdm
import torch
import random

set_seeds()

# Set up logger
logger = setup_logger("data_preprocessing", "logs/data_preprocessing.log")

data_visualizer = ModelVisualizer(save_dir="visualizations/data_analysis")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stock data preprocessing script")
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
        "--cache-dir", type=str, default="data/cache", help="Directory for cached data"
    )
    parser.add_argument(
        "--use-smote", action="store_true", help="Apply SMOTE for class balancing"
    )
    return parser.parse_args()


def get_cache_path(ticker, start_date, end_date, cache_dir="data/cache"):
    """Generate cache file path based on parameters."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_name = f"{ticker}_{start_date}_{end_date}.pkl"
    return cache_dir / cache_name


def get_sp500_tickers(top_n=None):
    """Get list of S&P 500 tickers from Wikipedia.

    Args:
        top_n (int, optional): Number of top companies to return).
                             If None, returns all companies.
    """
    logger.info("Retrieving S&P 500 tickers from Wikipedia")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500 = tables[0]  # First table contains S&P 500 companies

    if top_n is not None:

        # Get market cap data for sorting
        tickers = sp500["Symbol"].tolist()
        market_caps = {}

        logger.info(f"Fetching market cap data for {len(tickers)} companies")
        for ticker in tqdm(tickers, desc="Fetching market caps"):
            try:
                stock = yf.Ticker(ticker)
                market_cap = stock.info.get("marketCap", 0)
                market_caps[ticker] = market_cap
            except Exception as e:
                logger.warning(f"Failed to fetch market cap for {ticker}: {e}")
                market_caps[ticker] = 0

        # Sort tickers by market cap and take top N
        sorted_tickers = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
        return [ticker for ticker, _ in sorted_tickers[:top_n]]

    return sp500["Symbol"].tolist()


def load_stock_data(
    start_date="2015-01-01",
    end_date="2024-01-31",
    ticker="AAPL",
    cache_path=None,
):
    """Download stock data from Yahoo Finance or load from cache."""

    if cache_path and os.path.exists(cache_path):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Loading cached data from {cache_path}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Loaded {len(data)} stocks from cache")
        return data

    logger.info(f"Downloading {ticker} stock data from {start_date} to {end_date}")
    if ticker.startswith("SP500"):
        if ticker.replace("SP500", "") != "" and "TOP" in ticker:
            top_n = int(ticker.replace("SP500TOP", ""))
        else:
            top_n = None
        tickers = get_sp500_tickers(top_n)
        logger.info(f"Downloading data for {len(tickers)} S&P 500 stocks")
        all_data = {}
        for tick in tickers:
            try:
                stock_data = yf.download(tick, start=start_date, end=end_date)
                if not stock_data.empty:
                    all_data[tick] = stock_data
                    logger.info(f"Successfully downloaded data for {tick}")
            except Exception as e:
                logger.error(f"Error downloading {tick}: {str(e)}")
        pickle.dump(all_data, open(cache_path, "wb"))
        return all_data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    # convert to dict
    stock_data = {ticker: stock_data}
    if cache_path:
        logger.info(f"Caching data to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(stock_data, f)

    return stock_data


def calculate_daily_returns(df):
    """Calculate daily percentage returns based on adjusted closing price.

    Formula: Daily_Return = (Adj Close_t - Adj Close_t-1) / Adj Close_t-1 * 100
    """
    logger.info("Calculating daily returns")
    # Calculate returns using the specified formula
    df["Daily_Return"] = (
        (df["Adj Close"] - df["Adj Close"].shift(1)) / df["Adj Close"].shift(1) * 100
    )
    # Handle any missing values (due to holidays or weekends)
    df["Daily_Return"] = df["Daily_Return"].fillna(0).shift(1)

    return df


def create_extreme_events(df, threshold=2.0):
    """Create binary column for extreme events based on daily returns."""
    logger.info(f"Creating extreme events column with threshold {threshold}%")
    df["Extreme_Event"] = ((df["Daily_Return"].abs() > threshold)).astype(int)
    df["Extreme_Event"] = df["Extreme_Event"].shift(-1)
    return df


def perform_pca_analysis(X, n_components=None, variance_threshold=0.95):
    """Perform PCA analysis on the feature set.

    Args:
        X (np.ndarray): Input features
        n_components (int, optional): Number of components to keep. If None, use variance_threshold
        variance_threshold (float): Minimum explained variance to maintain (default: 0.95)

    Returns:
        tuple: (transformed_X, pca_model, explained_variance_ratio)
    """
    logger.info("Performing PCA analysis")

    # Handle 3D sequences by reshaping
    original_shape = X.shape
    is_3d = len(original_shape) == 3

    if is_3d:
        # Reshape to 2D: combine batch and sequence dimensions
        X = X.reshape(-1, original_shape[1])

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # If n_components not specified, use enough components to explain variance_threshold
    if n_components is None:
        n_components = min(X_scaled.shape)

    # Initialize and fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Calculate cumulative explained variance
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    # Find optimal number of components based on variance threshold
    n_optimal = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1

    # Plot explained variance
    data_visualizer.plot_pca_variance(
        pca.explained_variance_ratio_, cumulative_variance_ratio, n_optimal, save=True
    )

    logger.info(f"Optimal number of components: {n_optimal}")
    logger.info(
        f"Explained variance with {n_optimal} components: "
        f"{cumulative_variance_ratio[n_optimal-1]:.2%}"
    )

    # Return transformed data with optimal components
    return X_pca[:, :n_optimal], pca, pca.explained_variance_ratio_


def prepare_features(df, apply_pca=True):
    """Prepare feature set and handle missing values."""
    logger.info("Preparing features and handling missing values")

    # First get the columns we want
    features = [col for col in df.columns if col != "Extreme_Event"]

    # Create X and y before handling missing values
    X = df[features]
    y = df["Extreme_Event"]

    # Fill NA values with 0 instead of dropping them
    X = X.fillna(0)
    y = y.fillna(0)

    assert len(X) == len(y), "X and y must have the same length"
    return X, y


def split_time_series_data(X, y, train_size=0.7, val_size=0.15):
    """Split data into train, validation, and test sets preserving time order across all series.

    Args:
        X (np.ndarray): Input features with shape [n_samples, n_features]
        y (np.ndarray): Target labels with shape [n_samples]
        train_size (float): Proportion of data for training
        val_size (float): Proportion of data for validation

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Splitting data into train, validation, and test sets")
    # Determine the most common length
    lengths = [len(x) for x in X]
    majority_length = max(set(lengths), key=lengths.count)

    # Filter arrays to keep only those with the majority length
    y = [y[i] for i in range(len(y)) if len(X[i]) == majority_length]

    X = [x for x in X if len(x) == majority_length]

    # Calculate split indices, all should have the same length
    total_samples = len(X[0])
    train_idx = int(total_samples * train_size)
    val_idx = train_idx + int(total_samples * val_size)

    all_X_train = []
    all_X_val = []
    all_X_test = []
    all_y_train = []
    all_y_val = []
    all_y_test = []
    for i in range(len(X)):
        # Split the data using indices
        X_train = X[i][:train_idx]
        X_val = X[i][train_idx:val_idx]
        X_test = X[i][val_idx:]

        y_train = y[i][:train_idx]
        y_val = y[i][train_idx:val_idx]
        y_test = y[i][val_idx:]

        all_X_train.append(X_train)
        all_X_val.append(X_val)
        all_X_test.append(X_test)

        all_y_train.append(y_train)
        all_y_val.append(y_val)
        all_y_test.append(y_test)

    logger.info(
        f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}"
    )

    return all_X_train, all_X_val, all_X_test, all_y_train, all_y_val, all_y_test


def add_noise_to_sequence(batched_sequence, p=0.3):
    """
    Randomly apply noise to all points by adding or substracting a little
    Determine the magnitude of the noise based on the standard deviation of the
    sequence.
    """
    if random.random() > p:
        return batched_sequence
    for sequence in batched_sequence:
        for feature in range(sequence.shape[0]):
            noise_magnitude = torch.std(sequence[feature]) * torch.FloatTensor(
                1
            ).uniform_(0.9, 1.1).to(sequence.device)
            noise = torch.normal(
                mean=0.0, std=float(noise_magnitude), size=sequence[feature].shape
            ).to(sequence.device)
            sequence[feature] += noise
    return batched_sequence


def mask_sequence_augmentation(batched_sequence, p=0.1):
    """Mask a random set of points in each sequence."""
    if random.random() > p:
        return batched_sequence
    for sequence in batched_sequence:
        for feature in range(sequence.shape[0]):
            mask = np.random.choice([0, 1], size=sequence.shape[1], p=[0.9, 0.1])
            mask = torch.tensor(mask).to(sequence.device)
            sequence[feature] *= mask
    return batched_sequence


def apply_smote_to_sequences(X, y):
    """Apply SMOTE to balance the dataset with sequence data.

    Args:
        X (np.ndarray): Input features of shape [n_samples, n_features, sequence_length]
                       or [n_samples, n_features]
        y (np.ndarray): Target labels

    Returns:
        tuple: (X_resampled, y_resampled) as numpy arrays
    """
    logger.info("Original dataset shape: %s", Counter(y))

    # Check if input is 3D or 2D
    original_shape = X.shape
    is_3d = len(original_shape) == 3

    if is_3d:
        # Flatten 3D sequences
        n_samples, n_features, sequence_length = original_shape
        X_flat = X.reshape(n_samples, -1)
    else:
        # Already 2D, no need to flatten
        X_flat = X

    # Initialize SMOTE
    smote = SMOTE(random_state=SEED)

    # Fit and apply SMOTE
    X_resampled_flat, y_resampled = smote.fit_resample(X_flat, y)

    # Reshape back to original shape if input was 3D
    if is_3d:
        X_resampled = X_resampled_flat.reshape(-1, n_features, sequence_length)
    else:
        X_resampled = X_resampled_flat

    logger.info("Resampled dataset shape: %s", Counter(y_resampled))

    return X_resampled, y_resampled


def prepare_all_features(all_data, threshold=2.0, plot=False):
    """Prepare features for SP500 stocks and return them as separate items."""
    logger.info("Preparing features for SP500 stocks")
    all_features = []
    all_labels = []
    for ticker, df in tqdm(all_data.items(), desc="Processing stocks"):
        try:
            df = (
                df.droplevel(level=1, axis=1)
                if isinstance(df.columns, pd.MultiIndex)
                else df
            )

            # Calculate returns and extreme events for each stock
            df = calculate_daily_returns(df)
            df = create_extreme_events(df, threshold)
            df = df[
                [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "Daily_Return",
                    "Extreme_Event",
                ]
            ]
            if plot:
                data_visualizer.plot_extreme_events(ticker, df, threshold, save=True)
            X, y = prepare_features(df)

            all_features.append(np.array(X, dtype=np.float32))
            all_labels.append(np.array(y, dtype=np.int32))
            logger.info(f"Successfully processed {ticker}")
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            continue

    return all_features, all_labels


def create_sequences(
    all_X_data,
    all_y_data,
    sequence_length=10,
    use_smote=False,
    as_tensor=True,
    flatten=False,
    idx_to_keep=None,
):
    """Create sequences for temporal data for all possible points.

    Args:
        all_X_data (List[np.ndarray]): List of numpy arrays with shape [n_samples, n_features]
        all_y_data (List[np.ndarray]): List of numpy arrays with shape [n_samples]
        sequence_length (int): Length of each sequence
        as_tensor (bool): Whether to return PyTorch tensors or numpy arrays
        flatten_by (int, optional): Index to flatten sequences by

    Returns:
        tuple: (sequences, targets) either as PyTorch tensors or numpy arrays
        sequences shape: [n_sequences, n_features, sequence_length]
        targets shape: [n_sequences]
    """
    all_sequences = []
    all_targets = []
    for arr, y in zip(all_X_data, all_y_data):
        if len(arr) <= sequence_length:
            raise ValueError(
                f"Input length {len(arr)} must be greater than sequence_length {sequence_length}"
            )

        # Create sequences for all possible points
        sequences = []
        targets = []

        # Create a sequence for every point that has enough history
        for i in range(sequence_length, len(arr)):
            seq = arr[i - sequence_length : i]
            # remove the last point as target
            targets.append(y[i - 1])
            sequences.append(seq)

        # Convert to numpy arrays
        sequences = np.array(sequences)
        targets = np.array(targets)
        all_sequences.append(sequences)
        all_targets.append(targets)
    all_sequences = np.concatenate(np.array(all_sequences))
    all_targets = np.concatenate(np.array(all_targets))
    # Transpose sequences to shape [n_sequences, n_features, sequence_length] for CNN input
    all_sequences = all_sequences.transpose(0, 2, 1)
    if idx_to_keep:
        all_sequences = all_sequences[:, idx_to_keep, :]
    if flatten:
        # keep the idx provided
        all_sequences = all_sequences.reshape(all_sequences.shape[0], -1)
    if as_tensor:
        return torch.FloatTensor(all_sequences), torch.LongTensor(all_targets)
    if use_smote:
        logger.info("Applying SMOTE to sequences")
        all_sequences, all_targets = apply_smote_to_sequences(
            all_sequences, all_targets
        )
    return all_sequences, all_targets


def get_data(
    ticker="AAPL",
    start_date="2015-01-01",
    end_date="2024-01-31",
    threshold=2.0,
    use_smote=False,
    cache_dir="data/cache",
    sequence_length=10,
    as_tensor=False,
    flatten=False,
    apply_pca=False,
    idx_to_keep=None,
):
    """Fetch stocks, prepare features, and split into train, val, and test sets with sequences."""
    cache_path = get_cache_path(ticker, start_date, end_date, cache_dir)

    # Load data
    all_data = load_stock_data(
        start_date=start_date,
        end_date=end_date,
        ticker=ticker,
        cache_path=cache_path,
    )

    logger.info(f"Successfully downloaded data for {len(all_data)} stocks")
    all_features, all_labels = prepare_all_features(all_data, threshold, plot=True)

    # Split by sequence
    logger.info("Splitting data by sequence")
    (
        all_X_train,
        all_X_val,
        all_X_test,
        all_y_train,
        all_y_val,
        all_y_test,
    ) = split_time_series_data(
        all_features,
        all_labels,
    )

    # Create sequences for each split
    logger.info("Creating sequences for each split")
    X_train_seq, y_train_seq = create_sequences(
        all_X_train,
        all_y_train,
        sequence_length,
        as_tensor=as_tensor,
        flatten=flatten,
        idx_to_keep=idx_to_keep,
        use_smote=use_smote,
    )
    X_val_seq, y_val_seq = create_sequences(
        all_X_val,
        all_y_val,
        sequence_length,
        as_tensor=as_tensor,
        flatten=flatten,
        idx_to_keep=idx_to_keep,
        use_smote=use_smote,
    )

    X_test_seq, y_test_seq = create_sequences(
        all_X_test,
        all_y_test,
        sequence_length,
        as_tensor=as_tensor,
        flatten=flatten,
        idx_to_keep=idx_to_keep,
        use_smote=use_smote,
    )
    if apply_pca:
        logger.info("Applying PCA to sequences")
        variance_threshold = 0.95
        _, pca_model, variance_ratio = perform_pca_analysis(
            X_train_seq, variance_threshold=variance_threshold
        )
        # Calculate cumulative sum of variance ratios
        cumulative_variance_ratio = np.cumsum(variance_ratio)
        kept_idx = cumulative_variance_ratio >= variance_threshold
        X_train_seq = X_train_seq[:, kept_idx, :]
        X_val_seq = X_val_seq[:, kept_idx, :]
        X_test_seq = X_test_seq[:, kept_idx, :]
        logger.info(
            f"Explained variance with {len(kept_idx)} components: "
            f"{cumulative_variance_ratio[len(kept_idx)-1]:.2%}"
        )
    if any(len(arr) == 0 for arr in [X_train_seq, X_val_seq, X_test_seq]):
        raise ValueError("Not enough data to create sequences in one or more splits")

    return X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_data()
