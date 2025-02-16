#!/usr/bin/env python
"""
energy_samformer.py

This script downloads hourly energy consumption data for a given year,
constructs sliding-window samples, trains a SAMFormer model, evaluates the model
(using RMSE and MASE metrics), and plots sample predictions.
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


def get_consumption_data_year(year):
    """
    Downloads daily consumption data for a given year, concatenates all days into a single DataFrame,
    and performs basic preprocessing (e.g., dropping NaN values).

    Parameters:
        year (int): The year for which to retrieve data.

    Returns:
        np.ndarray: A 1-D array of float values extracted from the second column.
    """
    all_data = []
    current_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)

    while current_date <= end_date:
        date_str = f"{current_date.day}.{current_date.month}.{current_date.year}"
        url = f"https://www.eview.de/e1/p3Export.php?frame=StadtMS&p=0005;S~00000936;dg1;t{date_str}"
        print(f"Fetching data for {date_str} from:\n{url}")
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad status codes
            daily_df = pd.read_csv(StringIO(response.text), sep=';')
            all_data.append(daily_df)
        except Exception as e:
            print(f"Error fetching data for {date_str}: {e}")
        current_date += timedelta(days=1)

    combined_data = pd.concat(all_data, ignore_index=True)
    string_kws = combined_data.iloc[:, 1].values
    values_float = np.array([float(w.replace(',', '')) for w in string_kws])
    return values_float


def construct_sliding_window_data(data, seq_len, pred_len, time_increment=1):
    """
    Construct sliding window samples.

    Args:
      data (np.ndarray): A 2-D numpy array of shape (T, n_features).
      seq_len (int): The length of the historical (input) window.
      pred_len (int): The number of future time steps to predict.
      time_increment (int): Step size for the sliding window.

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


def read_library_energy_dataset(seq_len, pred_len, year):
    """
    Load energy consumption data for a given year, split it into train/test sets,
    and apply a sliding window.

    Args:
      seq_len (int): Length of historical context.
      pred_len (int): Forecast horizon length.
      year (int): Year for which to fetch data.

    Returns:
      Tuple: ((x_train, y_train), (x_test, y_test))
    """
    ts = get_consumption_data_year(year)
    ts = np.array(ts).reshape(-1, 1)
    n = len(ts)

    train_end = int(n * 0.6)
    train_series = ts[:train_end]
    test_series = ts[train_end - seq_len:]

    # (Optional) Uncomment below if you want to normalize the data.
    # scaler = StandardScaler()
    # scaler.fit(train_series)
    # train_series = scaler.transform(train_series)
    # test_series = scaler.transform(test_series)

    x_train, y_train = construct_sliding_window_data(train_series, seq_len, pred_len)
    x_test, y_test = construct_sliding_window_data(test_series, seq_len, pred_len)

    # Flatten targets if SAMFormer expects 2D targets.
    flatten = lambda y: y.reshape((y.shape[0], y.shape[1] * y.shape[2]))
    y_train, y_test = flatten(y_train), flatten(y_test)

    return (x_train, y_train), (x_test, y_test)


def plot_weather_data(weather_values, title="Weather Data"):
    """
    Plot energy consumption data from a 1-D array.

    Args:
      weather_values (np.ndarray): A 1-D array of energy consumption values.
      title (str): Title for the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(weather_values, label="Hourly Energy Consumption Rate Stadtbücherei", color="blue", linewidth=2)
    plt.xlabel("Time (Hours)")
    plt.ylabel("kWh")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot(x_context, y_true, y_pred, title="Predictions vs Ground Truth"):
    """
    Plot historical data, ground truth, and predicted values.

    Args:
      x_context (np.ndarray): Historical data array of shape (seq_len,).
      y_true (np.ndarray): Ground truth future values array of shape (pred_len,).
      y_pred (np.ndarray): Predicted future values array of shape (pred_len,).
      title (str): Title for the plot.
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
    # Specify parameters
    seq_len = 256    # Historical context length
    pred_len = 128   # Forecast horizon length
    year = 2024      # Year to evaluate
    batch_size = 64

    # (Optional) Plot raw consumption data
    # raw_values = get_consumption_data_year(year)
    # plot_weather_data(raw_values, title="Stadtbücherei Energy Consumption")

    # Read dataset and construct sliding window samples
    (x_train, y_train), (x_test, y_test) = read_library_energy_dataset(seq_len, pred_len, year)

    # Instantiate and train SAMFormer model
    model = SAMFormer(num_epochs=100,
                      batch_size=batch_size,
                      base_optimizer=torch.optim.Adam,
                      learning_rate=1e-3,
                      weight_decay=1e-5,
                      rho=0.5,
                      use_revin=True)

    print("Training SAMFormer model...")
    model.fit(x_train, y_train)

    # Evaluate the model on the test set
    print("Evaluating model...")
    y_pred_test = model.predict(x_test)
    rmse_val = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
    mase_val = mase(y_test, y_pred_test)
    print("RMSE:", rmse_val)
    print("MASE:", mase_val)

    # Plot a sample prediction from the test set
    sample_idx = 310  # Change index if needed
    x_context_sample = x_test[sample_idx].squeeze()  # shape: (seq_len,)
    y_true_sample = y_test[sample_idx].reshape(1, -1)[0]  # shape: (pred_len,)
    y_pred_sample = y_pred_test[sample_idx].reshape(1, -1)[0]  # shape: (pred_len,)
    plot(x_context_sample, y_true_sample, y_pred_sample, title="Energy Forecast Sample")


if __name__ == "__main__":
    main()
