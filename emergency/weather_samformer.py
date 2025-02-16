#!/usr/bin/env python
"""
weather_samformer.py

This script loads weather data from a CSV file, constructs sliding-window samples,
trains a SAMFormer model on the data, evaluates its predictions (computing RMSE and MASE),
and saves a sample prediction plot and evaluation metrics.
All outputs are saved under /content/results/weather.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import kagglehub
import requests
from io import StringIO
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from samformer import SAMFormer

# Set the results directory to /content/results/weather and ensure it exists
RESULTS_DIR = "/content/results/weather"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_weather_data(city: str) -> ArrayLike:
    """
    Downloads weather data for a given city from the Kaggle dataset.
    Returns a 1-D numpy array of daily average temperatures.
    """
    path = kagglehub.dataset_download("gucci1337/weather-of-albania-last-three-years")
    years = [2021, 2022, 2023]
    data_frames = []
    for year in years:
        file_path = f"{path}/data_weather/{city}/{city}{year}.csv"
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['tavg'])  # Remove rows where 'tavg' is NaN
        data_frames.append(df['tavg'])
    concatenated_data = pd.concat(data_frames, ignore_index=True)
    return concatenated_data.values

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

def read_weather_dataset(seq_len, pred_len, city):
    """
    Load weather data for a given city, split it into train/test sets,
    standardize using the training set, and apply a sliding window.
    """
    ts = get_weather_data(city)
    ts = np.array(ts).reshape(-1, 1)
    n = len(ts)
    train_end = int(n * 0.6)
    train_series = ts[:train_end]
    test_series = ts[train_end - seq_len:]
    scaler = StandardScaler()
    scaler.fit(train_series)
    train_series = scaler.transform(train_series)
    test_series = scaler.transform(test_series)
    x_train, y_train = construct_sliding_window_data(train_series, seq_len, pred_len)
    x_test, y_test = construct_sliding_window_data(test_series, seq_len, pred_len)
    # Flatten the targets if SAMFormer expects 2D targets
    flatten = lambda y: y.reshape((y.shape[0], y.shape[1] * y.shape[2]))
    y_train, y_test = flatten(y_train), flatten(y_test)
    return (x_train, y_train), (x_test, y_test)

def plot_weather_data(weather_values: ArrayLike, title: str = "Weather Data"):
    """
    Plot weather data from a 1-D array and save the figure to the results directory.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(weather_values, label="Daily Average Temperature", color="blue", linewidth=2)
    plt.xlabel("Time (Days)")
    plt.ylabel("Temperature")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(RESULTS_DIR, "weather_data.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Weather data plot saved to {save_path}")

def plot_weather_predictions(x_context, y_true, y_pred, title="Predictions vs Ground Truth"):
    """
    Plot historical data, ground truth, and predicted values and save to the results directory.
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
    save_path = os.path.join(RESULTS_DIR, "predictions_vs_ground_truth.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Prediction plot saved to {save_path}")

def mase(y_true, y_pred):
    """
    Compute the Mean Absolute Scaled Error (MASE).
    """
    mae_model = np.mean(np.abs(y_pred - y_true))
    naive_mae = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    return mae_model / naive_mae if naive_mae != 0 else float("inf")

def save_metrics_to_file(metrics: dict, filepath: str):
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filepath}")

def main():
    # Example: Get weather data and save its plot
    weather_values = get_weather_data("lezhe")
    plot_weather_data(weather_values, title="Weather Data for Lezhe")

    # Specify parameters
    seq_len = 256    # historical context length
    pred_len = 128   # forecast horizon length
    city = "lezhe"
    batch_size = 64

    # Read the weather dataset
    (x_train, y_train), (x_test, y_test) = read_weather_dataset(seq_len, pred_len, city)

    # Instantiate and train SAMFormer on the weather data
    model = SAMFormer(num_epochs=100,
                      batch_size=batch_size,
                      base_optimizer=torch.optim.Adam,
                      learning_rate=1e-3,
                      weight_decay=1e-5,
                      rho=0.5,
                      use_revin=True)
    model.fit(x_train, y_train)

    # Evaluate the model on the test set
    y_pred_test = model.predict(x_test)
    rmse_val = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
    mase_val = mase(y_test, y_pred_test)
    print('RMSE:', rmse_val)
    print('MASE:', mase_val)

    # Save metrics to a JSON file in the results directory
    metrics = {"RMSE": float(rmse_val), "MASE": float(mase_val)}
    metrics_filepath = os.path.join(RESULTS_DIR, "metrics.json")
    save_metrics_to_file(metrics, metrics_filepath)

    # Plot sample prediction for a chosen sample and save the plot
    sample_idx = 310  # choose a sample index
    x_context_sample = x_test[sample_idx][0]  # univariate: take the first channel
    y_true_sample = y_test[sample_idx].reshape(1, -1)[0]
    y_pred_sample = y_pred_test[sample_idx].reshape(1, -1)[0]
    plot_weather_predictions(x_context_sample, y_true_sample, y_pred_sample,
                             title="Weather Forecast for City 'Lezhe'")

if __name__ == "__main__":
    main()
