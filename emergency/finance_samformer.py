#!/usr/bin/env python
"""
amzn_samformer.py

This script loads financial time-series data from a CSV file,
constructs sliding-window samples, trains a SAMFormer model,
evaluates its predictions (computing RMSE and MAE),
and produces plots comparing ground truth and predictions.
The plots and metrics are saved to /content/results/finance.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from io import StringIO
from sklearn.preprocessing import MinMaxScaler  # Changed from StandardScaler
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
    normalize using the training set with a MinMaxScaler, and apply a sliding window.

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
    
    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
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


def plot_data(values, output_path, title="Data"):
    """
    Plot a 1-D time series and save the plot to the specified file.

    Args:
      values: A 1-D numpy array containing the data.
      output_path: Absolute path (including filename) where the plot is saved.
      title: Title for the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(values, label="Time-series data", color="blue", linewidth=2)
    plt.xlabel("Time Steps")
    plt.ylabel("Series")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Raw data plot saved to {output_path}")


def plot(x_context, y_true, y_pred, output_path, title="Predictions vs Ground Truth"):
    """
    Plot historical data, ground truth, and predicted future values, then save the plot.

    Args:
      x_context: Historical data array of shape (seq_len,).
      y_true: Ground truth future values array of shape (pred_len,).
      y_pred: Predicted future values array of shape (pred_len,).
      output_path: Absolute path (including filename) to save the plot.
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
    plt.savefig(output_path)
    plt.close()
    print(f"Prediction plot saved to {output_path}")


def main():
    # Set output directory for plots and metrics
    output_dir = "/content/results/finance"
    os.makedirs(output_dir, exist_ok=True)

    # Plot and save the raw data
    raw_values = get_data()
    raw_data_plot_path = os.path.join(output_dir, "raw_data.png")
    print("Plotting raw financial time-series data...")
    plot_data(raw_values, raw_data_plot_path, title="AMZN Stock Price Data")
    
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
    mae_val = np.mean(np.abs(y_test - y_pred_test))
    
    print("RMSE:", rmse_val)
    print("MAE:", mae_val)
    
    # Save metrics to a JSON file
    metrics = {"RMSE": float(rmse_val), "MAE": float(mae_val)}
    metrics_filepath = os.path.join(output_dir, "metrics.json")
    with open(metrics_filepath, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_filepath}")
    
    # Plot sample prediction for a chosen sample (here sample index 310)
    sample_idx = 310
    x_context_sample = x_test[sample_idx].squeeze()  # shape: (seq_len,)
    y_true_sample = y_test[sample_idx].reshape(1, -1)[0]  # shape: (pred_len,)
    y_pred_sample = y_pred_test[sample_idx].reshape(1, -1)[0]  # shape: (pred_len,)
    pred_plot_path = os.path.join(output_dir, "sample_prediction.png")
    plot(x_context_sample, y_true_sample, y_pred_sample, pred_plot_path,
         title="Ground Truth vs Predictions")
    

if __name__ == "__main__":
    main()
