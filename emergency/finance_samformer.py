#!/usr/bin/env python
"""
amzn_samformer.py

This script loads financial time-series data from a CSV file,
constructs sliding-window samples, trains a SAMFormer model,
evaluates its predictions (computing RMSE and MASE),
and produces plots comparing ground truth and predictions.
"""

import torch
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from io import StringIO
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import the SAMFormer model from the samformer package.
from samformer import SAMFormer


def get_data():
    """
    Returns a numpy array containing the column with the actual data from a CSV file.
    
    The CSV file is assumed to be located at "/content/AMZN-stock-price.csv".
    """
    CSV_FILE_ABSOLUTE_PATH = "/content/AMZN-stock-price.csv"
    df = pd.read_csv(CSV_FILE_ABSOLUTE_PATH)
    # Assume the second column holds the desired data.
    df = df.iloc[:, 1]
    return df.values


def construct_sliding_window_data(data, seq_len, pred_len, time_increment=1):
    """
    Construct sliding window samples.

    Args:
      data: A 2-D numpy array of shape (T, n_features)
      seq_len: The length of the historical (input) window.
      pred_len: The number of future time steps to predict.
      time_increment: Step size for the sliding window.

    Returns:
      Tuple (x, y) where:
        - x is of shape (n_samples, n_features, seq_len)
        - y is of shape (n_samples, n_features, pred_len)
    """
    n_samples = data.shape[0] - (seq_len - 1) - pred_len
    indices = np.arange(0, n_samples, time_increment)
    x, y = [], []
    for i in indices:
        # Transpose so that each sample becomes (n_features, seq_len)
        x.append(data[i:(i + seq_len)].T)
        y.append(data[(i + seq_len):(i + seq_len + pred_len)].T)
    return np.array(x), np.array(y)


def train_test_split(seq_len, pred_len):
    """
    Load the data from CSV, split it into train and test sets,
    standardize using the training set, and apply a sliding window.

    Args:
      seq_len: Length of historical context.
      pred_len: Forecast horizon length.

    Returns:
      Tuple: ((x_train, y_train), (x_test, y_test))
    """
    ts = get_data()
    ts = np.array(ts).reshape(-1, 1)  # Ensure 2-D array (for univariate data)
    n = len(ts)
    
    # 60-40 split
    train_end = int(n * 0.6)
    train_series = ts[:train_end]
    # To allow full sliding-window samples in test, include last seq_len points of train.
    test_series = ts[train_end - seq_len:]
    
    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_series)
    train_series = scaler.transform(train_series)
    test_series = scaler.transform(test_series)
    
    # Construct sliding-window samples
    x_train, y_train = construct_sliding_window_data(train_series, seq_len, pred_len)
    x_test, y_test = construct_sliding_window_data(test_series, seq_len, pred_len)
    
    # Flatten targets if SAMFormer expects 2D targets
    flatten = lambda y: y.reshape((y.shape[0], y.shape[1] * y.shape[2]))
    y_train, y_test = flatten(y_train), flatten(y_test)
    
    return (x_train, y_train), (x_test, y_test)


def plot_data(values, title="Data"):
    """
    Plot a 1-D time series and display the plot.

    Args:
      values: A 1-D numpy array containing the data.
      title: Title for the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(values, label="Time-series data", color="blue", linewidth=2)
    plt.xlabel("Time Steps")
    plt.ylabel("Series")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot(x_context, y_true, y_pred, title="Predictions vs Ground Truth"):
    """
    Plot historical data, ground truth, and predicted future values.

    Args:
      x_context: Historical data array of shape (seq_len,).
      y_true: Ground truth future values array of shape (pred_len,).
      y_pred: Predicted future values array of shape (pred_len,).
      title: Title for the plot.
    """
    seq_len = len(x_context)
    pred_len = len(y_true)
    
    plt.figure(figsize=(12, 6))
    # Plot historical data
    plt.plot(range(seq_len), x_context, label="Historical Data", color="blue", linewidth=2)
    # Plot ground truth
    plt.plot(range(seq_len, seq_len + pred_len),
             y_true, label="Ground Truth", color="green", linestyle="--", linewidth=2)
    # Plot predictions
    plt.plot(range(seq_len, seq_len + pred_len),
             y_pred, label="Prediction", color="red", linewidth=2)
    
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def mase(y_true, y_pred):
    """
    Compute the Mean Absolute Scaled Error (MASE).

    Args:
      y_true: Ground truth values.
      y_pred: Predicted values.

    Returns:
      MASE value.
    """
    mae_model = np.mean(np.abs(y_pred - y_true))
    naive_mae = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    return mae_model / naive_mae if naive_mae != 0 else float("inf")


def main():
    # Plot the raw data
    values = get_data()
    print("Plotting the raw time series data...")
    plot_data(values, title="AMZN Stock Price Data")
    
    # Specify parameters
    seq_len = 256    # historical context length
    pred_len = 128   # forecast horizon length
    batch_size = 64

    # Split data into train and test sets using sliding windows
    (x_train, y_train), (x_test, y_test) = train_test_split(seq_len, pred_len)

    # Instantiate and train SAMFormer on the training data
    print("Training SAMFormer model...")
    model = SAMFormer(num_epochs=100,
                      batch_size=batch_size,
                      base_optimizer=torch.optim.Adam,
                      learning_rate=1e-3,
                      weight_decay=1e-5,
                      rho=0.5,
                      use_revin=True)
    
    model.fit(x_train, y_train)
    
    # Evaluate the model on the test set
    print("Evaluating model on test data...")
    y_pred_test = model.predict(x_test)
    rmse_val = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
    mase_val = mase(y_test, y_pred_test)
    
    print('RMSE:', rmse_val)
    print('MASE:', mase_val)
    
    # Plot predictions for a chosen sample (e.g., sample index 310)
    sample_idx = 310
    x_context_sample = x_test[sample_idx].squeeze()  # shape: (seq_len,)
    # y_test is flattened; reshape to (1, pred_len) and take first row
    y_true_sample = y_test[sample_idx].reshape(1, -1)[0]
    y_pred_sample = y_pred_test[sample_idx].reshape(1, -1)[0]
    plot(x_context_sample, y_true_sample, y_pred_sample, title="Ground Truth vs Predictions")


if __name__ == "__main__":
    main()
