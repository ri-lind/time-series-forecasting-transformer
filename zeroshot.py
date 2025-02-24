import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
import kagglehub
import json
import os
import argparse
import requests
from io import StringIO
from datetime import datetime, timedelta
from transformers import AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  # Import StandardScaler

def get_weather_data(city: str) -> ArrayLike:
    path = kagglehub.dataset_download("gucci1337/weather-of-albania-last-three-years")
    years = [2021, 2022, 2023]
    data_frames = []
    for year in years:
        file_path = f"{path}/data_weather/{city}/{city}{year}.csv"
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['tavg'])  # Remove rows where 'tavg' is NaN
        data_frames.append(df['tavg'])
    concatenated_data = pd.concat(data_frames, ignore_index=True)
    
    # Scale the data using StandardScaler
    scaler = StandardScaler()
    data_reshaped = concatenated_data.values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(data_reshaped).flatten()
    
    return scaled_data

def get_finance_data():
    """
    Returns a numpy array containing the finance data.
    """
    CSV_FILE_ABSOLUTE_PATH = "/content/AMZN-stock-price.csv"
    df = pd.read_csv(CSV_FILE_ABSOLUTE_PATH)
    # Assuming the second column holds the desired data.
    df = df.iloc[:, 1]
    
    # Scale the data using StandardScaler
    scaler = StandardScaler()
    data = df.values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(data).flatten()
    
    return scaled_data

def get_consumption_data_year(year: int):
    """
    Downloads daily consumption data for a given year, concatenates all days into a single DataFrame,
    and performs basic preprocessing (e.g., dropping NaN values).
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
    string_values = combined_data.iloc[:, 1].values
    values_float = np.array([float(w.replace(',', '')) for w in string_values])
    
    # Scale the data using StandardScaler
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(values_float.reshape(-1, 1)).flatten()
    
    return scaled_values

def get_new_cases_by_country(df: pd.DataFrame) -> dict:
    """
    Groups the DataFrame by 'Country', sorts by 'Date_reported', and returns a dictionary 
    mapping country names to a NumPy array of new cases.
    """
    result = {}
    for country, group in df.groupby('Country'):
        group_sorted = group.sort_values('Date_reported')
        new_cases_array = group_sorted['New_cases'].to_numpy()
        result[country] = new_cases_array
    return result

def get_healthcare_data(country: str) -> ArrayLike:
    """
    Returns a NumPy array of new COVID-19 cases for the specified country.
    Data is taken from the WHO global daily dataset, and a slice [200:1200] is returned.
    """
    CSV_FILE_ABSOLUTE_PATH = "/content/WHO-COVID-19-global-daily-data.csv"
    df = pd.read_csv(CSV_FILE_ABSOLUTE_PATH)
    df = df.sort_values(['Country', 'Date_reported'])
    df['New_cases'] = df.groupby('Country')['New_cases'].transform(lambda group: group.interpolate(method='linear'))
    cases_dict = get_new_cases_by_country(df)
    new_cases = cases_dict.get(country)
    if new_cases is None:
        raise ValueError(f"Country '{country}' not found in dataset")
    
    sliced_cases = new_cases[200:1200]
    
    # Scale the data using StandardScaler
    scaler = StandardScaler()
    scaled_cases = scaler.fit_transform(sliced_cases.reshape(-1, 1)).flatten()
    
    return scaled_cases

# ---------------------------
# ZeroShotForecast Class (unchanged except for metric calculation)
# ---------------------------

class ZeroShotForecast:
    def __init__(self, data: np.ndarray, context_length: int = 256, prediction_length: int = 128,
                 model_name: str = 'Maple728/TimeMoE-50M', device: str = "cpu",
                 results_dir: str = "zeroshot_results"):
        """
        :param data: 1D numpy array containing the full time series.
        :param context_length: Number of points to use as context.
        :param prediction_length: Number of points to forecast.
        :param model_name: Hugging Face model identifier.
        :param device: "cpu" or "cuda" (if available).
        :param results_dir: Directory to save metrics and plots.
        """
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.model_name = model_name
        self.device = device
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.model = None
        self.training_data = None  # Will store training data for MASE computation

    def load_model(self):
        print(f"Loading model {self.model_name} on device {self.device} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,  # "cpu" or "cuda"
            trust_remote_code=True,
        )
        self.model.eval()

    def prepare_data(self):
        """
        Splits the full data into training (first 60%) and test (remaining 40%),
        then extracts the context (last context_length of training data) and
        the ground truth forecast (first prediction_length of test data).
        Also stores the training data for MASE computation.
        """
        sequence_list = self.data.tolist() if hasattr(self.data, "tolist") else list(self.data)
        split_idx = int(len(sequence_list) * 0.6)
        training_data = sequence_list[:split_idx]
        test_data = sequence_list[split_idx:]
        self.training_data = np.array(training_data, dtype=np.float32)
        
        if len(training_data) < self.context_length:
            raise ValueError("Not enough training data for the specified context length.")
        if len(test_data) < self.prediction_length:
            raise ValueError("Not enough test data for the specified prediction length.")

        context = training_data[-self.context_length:]  # last context_length points of training data
        ground_truth = test_data[:self.prediction_length]  # first prediction_length points of test data
        return np.array(context, dtype=np.float32), np.array(ground_truth, dtype=np.float32)

    def forecast(self):
        # Prepare the context and ground truth
        context, ground_truth = self.prepare_data()
        # Convert context to tensor with shape [1, context_length]
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        # Normalize the context along the last dimension (per sample)
        mean = context_tensor.mean(dim=-1, keepdim=True)
        std = context_tensor.std(dim=-1, keepdim=True)
        normed_context = (context_tensor - mean) / std

        # Forecast using the model
        print("Generating forecast ...")
        with torch.no_grad():
            output = self.model.generate(normed_context, max_new_tokens=self.prediction_length)
        # The output shape is [batch_size, context_length + prediction_length]
        normed_predictions = output[:, -self.prediction_length:]
        # Inverse normalization
        predictions = normed_predictions * std + mean
        # Convert predictions to a 1D numpy array
        predictions_np = predictions.squeeze(0).cpu().numpy()

        return predictions_np, ground_truth

    def calculate_metrics(self, predictions: np.ndarray, ground_truth: np.ndarray) -> dict:
        # Standard metrics: MAE and RMSE
        mae = np.mean(np.abs(predictions - ground_truth))
        mse = np.mean((predictions - ground_truth) ** 2)
        rmse = np.sqrt(mse)
        # Compute naive forecast error from training data (mean absolute difference)
        naive_error = np.mean(np.abs(np.diff(self.training_data)))
        mase = mae / (naive_error + 1e-8)
        return {"RMSE": float(rmse), "MAE": float(mae), "MASE": float(mase)}

    def plot_results(self, predictions: np.ndarray, ground_truth: np.ndarray) -> str:
        plt.figure(figsize=(10, 5))
        plt.plot(ground_truth, label="Ground Truth", marker="o")
        plt.plot(predictions, label="Forecast", marker="x")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title("Forecast vs. Ground Truth")
        plt.legend()
        plot_path = os.path.join(self.results_dir, "forecast_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")
        return plot_path

    def save_metrics(self, metrics: dict):
        metrics_path = os.path.join(self.results_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_path}")

    def run(self):
        self.load_model()
        predictions, ground_truth = self.forecast()
        metrics = self.calculate_metrics(predictions, ground_truth)
        self.save_metrics(metrics)
        self.plot_results(predictions, ground_truth)
        print("Zero-shot forecasting completed.")
        print("Metrics:", metrics)

# ---------------------------
# Main: Argument Parsing
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot Forecasting with TimeMoE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--finance", action="store_true", help="Fetch finance data")
    group.add_argument("-e", "--energy", action="store_true", help="Fetch energy data")
    group.add_argument("-c", "--city", help="Name of the city for weather data")
    group.add_argument("--healthcare", help="Name of the country for healthcare data")
    parser.add_argument("--year", type=int, default=2023, help="Year for energy data (default: 2023)")
    parser.add_argument("--device", default="cpu", help="Device to run the model on: 'cpu' or 'cuda'")
    args = parser.parse_args()

    # Fetch the data based on the provided argument
    if args.finance:
        print("Fetching finance data ...")
        data = get_finance_data()
    elif args.energy:
        print(f"Fetching energy data for year {args.year} ...")
        data = get_consumption_data_year(args.year)
    elif args.city:
        print(f"Fetching weather data for city {args.city} ...")
        data = get_weather_data(args.city)
    elif args.healthcare:
        print(f"Fetching healthcare data for country {args.healthcare} ...")
        data = get_healthcare_data(args.healthcare)
    else:
        raise ValueError("No valid data source argument provided.")

    # Instantiate and run the forecasting class
    
    print("Experiment with 64 - 32 split")
    forecast_engine = ZeroShotForecast(
        data=data,
        context_length=64,
        prediction_length=32,
        model_name='Maple728/TimeMoE-50M',
        device=args.device,
        results_dir="zeroshot_results_small"
    )
    forecast_engine.run()
    
    
    print("Experiment with 128 - 64 split")
    forecast_engine = ZeroShotForecast(
        data=data,
        context_length=128,
        prediction_length=64,
        model_name='Maple728/TimeMoE-50M',
        device=args.device,
        results_dir="zeroshot_results_medium"
    )
    forecast_engine.run()
    
    
    print("Experiment with 256 - 128 split")
    forecast_engine = ZeroShotForecast(
        data=data,
        context_length=256,
        prediction_length=128,
        model_name='Maple728/TimeMoE-50M',
        device=args.device,
        results_dir="zeroshot_results_large"
    )
    forecast_engine.run()

if __name__ == "__main__":
    main()
