#!/usr/bin/env python
"""
covid_samformer.py

This script loads COVID-19 new cases data from a CSV file, constructs sliding-window samples,
trains a SAMFormer model on the data, evaluates the model (computing RMSE and MASE), and produces
a plot comparing the ground truth and predictions.
"""

import torch
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from io import StringIO
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import SAMFormer from the samformer package
from samformer import SAMFormer


def get_new_cases_by_country(df: pd.DataFrame) -> dict:
    """
    Given a DataFrame with columns including 'Date_reported', 'Country', 'New_cases',
    returns a dictionary where keys are country names and values are NumPy arrays of new cases.
    """
    result = {}
    for country, group in df.groupby('Country'):
        group_sorted = group.sort_values('Date_reported')
        new_cases_array = group_sorted['New_cases'].to_numpy()
        result[country] = new_cases_array
    return result


def get_data():
    """
    Reads the WHO COVID-19 global daily data CSV, interpolates missing 'New_cases' values,
    and returns a 1-D numpy array for Germany (rows 200 to 1200).
    """
    CSV_FILE_ABSOLUTE_PATH = "/content/WHO-COVID-19-global-daily-data.csv"
    df = pd.read_csv(CSV_FILE_ABSOLUTE_PATH)
    df = df.sort_values(['Country', 'Date_reported'])
    # Interpolate missing New_cases per country
    df['New_cases'] = df.groupby('Country')['New_cases'].transform(lambda group: group.interpolate(method='linear'))
    cases_dict = get_new_cases_by_country(df)
    new_cases = cases_dict["Germany"][200:1200]
    return new_cases


def construct_sliding_window_data(data, seq_len, pred_len, time_increment=1):
    """
    Construct sliding window samples from a 2-D array.

    Args:
      data (np.ndarray): Array of shape (T, n_features)
      seq_len (int): Length of the input (historical) window.
      pred_len (int): Number of future steps to predict.
      time_increment (int): Step size between windows.

    Returns:
      Tuple (x, y) where:
        - x is of shape (n_samples, n_features, seq_len)
        - y is of shape (n_samples, n_features, pred_len)
    """
    n_samples = data.shape[0] - (seq_len - 1) - pred_len
    indices = np.arange(0, n_samples, time_increment)
    x, y = [], []
    for i in indices:
        x.append(data[i:(i + seq_len)].T)
        y.append(data[(i + seq_len):(i + seq_len + pred_len)].T)
    return np.array(x), np.array(y)


def train_test_split(seq_len, pred_len):
    """
    Load the COVID-19 new cases data, split it into training and testing sets,
    normalize using StandardScaler based on the training set, and generate sliding-window samples.

    Args:
      seq_len (int): Historical context length.
      pred_len (int): Forecast horizon length.

    Returns:
      Tuple: ((x_train, y_train), (x_test, y_test))
    """
    ts = get_data()
    ts = np.array(ts).reshape(-1, 1)
    n = len(ts)
    train_end = int(n * 0.6)
    train_series = ts[:train_end]
    test_series = ts[train_end - seq_len:]

    # Normalize data based on training set
    scaler = StandardScaler()
    scaler.fit(train_series)
    train_series = scaler.transform(train_series)
    test_series = scaler.transform(test_series)

    x_train, y_train = construct_sliding_window_data(train_series, seq_len, pred_len)
    x_test, y_test = construct_sliding_window_data(test_series, seq_len, pred_len)

    # Flatten targets if SAMFormer expects 2D targets
    flatten = lambda y: y.reshape((y.shape[0], y.shape[1] * y.shape[2]))
    y_train, y_test = flatten(y_train), flatten(y_test)

    return (x_train, y_train), (x_test, y_test)


def plot_data(values, title="Data"):
    """
    Plot a 1-D time series.

    Args:
      values (np.ndarray): 1-D array of data values.
      title (str): Plot title.
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
    Plot historical input, ground truth future values, and predictions.

    Args:
      x_context (np.ndarray): Array of shape (seq_len,) for historical data.
      y_true (np.ndarray): Array of shape (pred_len,) for ground truth.
      y_pred (np.ndarray): Array of shape (pred_len,) for predictions.
      title (str): Plot title.
    """
    seq_len = len(x_context)
    pred_len = len(y_true)

    plt.figure(figsize=(12, 6))
    plt.plot(range(seq_len), x_context, label="Historical Data", color="blue", linewidth=2)
    plt.plot(range(seq_len, seq_len + pred_len),
             y_true, label="Ground Truth", color="green", linestyle="--", linewidth=2)
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
      y_true (np.ndarray): Ground truth values.
      y_pred (np.ndarray): Predicted values.

    Returns:
      float: The MASE value.
    """
    mae_model = np.mean(np.abs(y_pred - y_true))
    naive_mae = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    return mae_model / naive_mae if naive_mae != 0 else float("inf")


def main():
    # Get and plot the raw data
    values = get_data()
    print("Plotting raw COVID-19 new cases data for Germany...")
    plot_data(values, title="Germany COVID-19 New Cases (200:1200)")

    # Set parameters
    seq_len = 256   # Historical context length
    pred_len = 128  # Forecast horizon length
    batch_size = 64

    # Generate train/test sliding-window samples
    (x_train, y_train), (x_test, y_test) = train_test_split(seq_len, pred_len)

    # Instantiate and train SAMFormer model
    print("Training SAMFormer model on COVID-19 data...")
    model = SAMFormer(num_epochs=100,
                      batch_size=batch_size,
                      base_optimizer=torch.optim.Adam,
                      learning_rate=1e-3,
                      weight_decay=1e-5,
                      rho=0.5,
                      use_revin=True)

    model.fit(x_train, y_train)

    # Evaluate model on test set
    print("Evaluating model on test set...")
    y_pred_test = model.predict(x_test)
    rmse_val = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
    mase_val = mase(y_test, y_pred_test)
    print("RMSE:", rmse_val)
    print("MASE:", mase_val)

    # Plot sample prediction (first sample in test set)
    sample_idx = 0
    x_context_sample = x_test[sample_idx].squeeze()   # shape: (seq_len,)
    y_true_sample = y_test[sample_idx].reshape(1, -1)[0]  # shape: (pred_len,)
    y_pred_sample = y_pred_test[sample_idx].reshape(1, -1)[0]  # shape: (pred_len,)
    plot(x_context_sample, y_true_sample, y_pred_sample, title="Ground Truth vs Predictions")


if __name__ == "__main__":
    main()
