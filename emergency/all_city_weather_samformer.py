#!/usr/bin/env python
"""
weather_samformer.py

This script downloads weather data for several cities, constructs sliding-window samples,
trains a SAMFormer model on the data, evaluates predictions using RMSE and MASE, and saves
plots and metrics to an output directory.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from samformer import SAMFormer
from sklearn.preprocessing import StandardScaler


def get_weather_data(city: str) -> ArrayLike:
    """
    Download weather data for a given city from the Kaggle dataset.
    
    Args:
      city (str): The name of the city.
      
    Returns:
      ArrayLike: A 1-D array of temperature values.
    """
    path = kagglehub.dataset_download("gucci1337/weather-of-albania-last-three-years")
    years = [2021, 2022, 2023]
    data_frames = []
    for year in years:
        file_path = f"{path}/data_weather/{city}/{city}{year}.csv"
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['tavg'])  # Remove rows with NaN in 'tavg'
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
    Load weather data for a given city, split into train/test, standardize using training set,
    and apply a sliding window.

    Args:
      seq_len (int): Length of historical context.
      pred_len (int): Forecast horizon length.
      city (str): City name used to fetch data.

    Returns:
      Tuple: ((x_train, y_train), (x_test, y_test))
    """
    ts = get_weather_data(city)
    ts = np.array(ts).reshape(-1, 1)  # Ensure 2-D array (univariate)
    n = len(ts)
    train_end = int(n * 0.6)
    train_series = ts[:train_end]
    # Include the last seq_len points from train for test split
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
    Plot weather data from a 1-D array and display it.
    
    Args:
      weather_values: A 1-D numpy array containing weather values.
      title (str): Title for the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(weather_values, label="Daily Average Temperature", color="blue", linewidth=2)
    plt.xlabel("Time (Days)")
    plt.ylabel("Temperature")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_weather_predictions(x_context, y_true, y_pred, save_path: str):
    """
    Plot historical data, ground truth, and predicted values, then save the figure.
    
    Args:
      x_context: Historical data array of shape (seq_len,).
      y_true: Ground truth future values array of shape (pred_len,).
      y_pred: Predicted future values array of shape (pred_len,).
      save_path (str): Path where the plot image will be saved.
    """
    title = "Predictions vs Ground Truth"
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
    plt.savefig(save_path)
    plt.show()


def mase_metric(y_true, y_pred):
    """
    Compute the Mean Absolute Scaled Error (MASE).

    Args:
      y_true: Ground truth array.
      y_pred: Predicted array.

    Returns:
      float: The MASE value.
    """
    mae_model = np.mean(np.abs(y_pred - y_true))
    naive_mae = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    return mae_model / naive_mae if naive_mae != 0 else float("inf")


def main():
    # Create an output directory for results
    output_dir = "SAMFormer_Results"
    os.makedirs(output_dir, exist_ok=True)

    # First, download and plot weather data for a single city example
    weather_values = get_weather_data("lezhe")
    print("Plotting weather data for Lezhe...")
    plot_weather_data(weather_values, title="Weather Data for Lezhe")

    # Define parameters
    seq_len = 256      # Historical context length
    pred_len = 128     # Forecast horizon length
    batch_size = 64

    # List of cities to process
    cities = ["elbasan", "gramsh", "himare", "kavaje", "korce",
              "kruje", "kukes", "lezhe", "sarande", "tirana", "vlore"]

    # Instantiate SAMFormer model
    model = SAMFormer(num_epochs=100,
                      batch_size=batch_size,
                      base_optimizer=torch.optim.Adam,
                      learning_rate=1e-3,
                      weight_decay=1e-5,
                      rho=0.5,
                      use_revin=True)

    # Train on each city and accumulate test data
    allx_test = []
    ally_test = []
    for city in cities:
        print(f"Processing city: {city}")
        (x_train, y_train), (x_test, y_test) = read_weather_dataset(seq_len, pred_len, city)
        # Train the model on the current city's training data
        model.fit(x_train, y_train)
        allx_test.append(x_test)
        ally_test.append(y_test)

    # Concatenate all test samples
    allx_test = np.concatenate(allx_test, axis=0)
    ally_test = np.concatenate(ally_test, axis=0)

    # Evaluate the model on the combined test set
    print("Predicting on the combined test set...")
    y_pred_test = model.predict(allx_test)
    rmse = np.sqrt(np.mean((ally_test - y_pred_test) ** 2))
    mase_val = mase_metric(ally_test, y_pred_test)

    print('RMSE:', rmse)
    print('MASE:', mase_val)

    # Save metrics to a JSON file
    metrics = {"RMSE": float(rmse), "MASE": float(mase_val)}
    metrics_filepath = os.path.join(output_dir, "metrics.json")
    with open(metrics_filepath, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_filepath}")

    # For each city, generate a sample prediction plot and save it
    for city in cities:
        print(f"Generating prediction plot for {city}...")
        (x_train, y_train), (x_test, y_test) = read_weather_dataset(seq_len, pred_len, city)
        y_pred_test_city = model.predict(x_test)
        # Take the first sample for plotting
        x_context_sample = x_test[0].squeeze()    # shape: (seq_len,)
        y_true_sample = y_test[0].squeeze()         # shape: (pred_len,)
        y_pred_sample = y_pred_test_city[0].squeeze()  # shape: (pred_len,)
        save_path = os.path.join(output_dir, f"{city}_sample_0.png")
        plot_weather_predictions(x_context_sample, y_true_sample, y_pred_sample, save_path)
        print(f"Prediction plot for {city} saved to {save_path}")


if __name__ == "__main__":
    main()
