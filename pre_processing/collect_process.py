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

def get_weather_data(city: str) -> ArrayLike:
    path = kagglehub.dataset_download("gucci1337/weather-of-albania-last-three-years")
    years = [2021, 2022, 2023]
    data_frames = []

    for year in years:
        file_path = f"{path}/data_weather/{city}/{city}{year}.csv"
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['tavg'])  # Remove rows where 'tavg' is NaN
        data_frames.append(df['tavg'])

    # Concatenate the 'tavg' columns from each year's DataFrame
    concatenated_data = pd.concat(data_frames, ignore_index=True)
    return concatenated_data.values

def get_finance_data():
    """
    Returns a numpy array containing the finance data.
    """
    CSV_FILE_ABSOLUTE_PATH = "/content/AMZN-stock-price.csv"
    df = pd.read_csv(CSV_FILE_ABSOLUTE_PATH)
    # Assuming the second column holds the desired data.
    df = df.iloc[:, 1]
    return df.values

def get_consumption_data_year(year: int):
    """
    Downloads daily consumption data for a given year, concatenates all days into a single DataFrame,
    and performs basic preprocessing (e.g., dropping NaN values).

    Parameters:
        year (int): The year for which to retrieve data.

    Returns:
        numpy.array: Array of preprocessed energy consumption values.
    """
    all_data = []
    # Define the start and end dates for the year
    current_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)

    while current_date <= end_date:
        # Format the date as day.month.year (e.g., "1.10.2023" for October 1, 2023)
        date_str = f"{current_date.day}.{current_date.month}.{current_date.year}"
        url = f"https://www.eview.de/e1/p3Export.php?frame=StadtMS&p=0005;S~00000936;dg1;t{date_str}"
        print(f"Fetching data for {date_str} from:\n{url}")

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad status codes

            # Read the CSV data from the response (assuming ';' as the delimiter)
            daily_df = pd.read_csv(StringIO(response.text), sep=';')
            all_data.append(daily_df)
        except Exception as e:
            print(f"Error fetching data for {date_str}: {e}")

        # Move to the next day
        current_date += timedelta(days=1)

    combined_data = pd.concat(all_data, ignore_index=True)
    # Assume the desired consumption data is in the second column with comma as decimal separator.
    string_values = combined_data.iloc[:, 1].values
    values_float = np.array([float(w.replace(',', '')) for w in string_values])
    return values_float

def main():
    # Setup command-line argument parser with mutually exclusive options
    parser = argparse.ArgumentParser(
        description="Fetch and save weather, finance, or energy data based on provided arguments."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--finance", action="store_true", help="Fetch finance data")
    group.add_argument("--energy", action="store_true", help="Fetch energy data")
    group.add_argument("-city", help="Name of the city for weather data")
    parser.add_argument("--year", type=int, default=2023, help="Year for energy data (default: 2023)")
    args = parser.parse_args()

    jsonl_dir = "/content/jsonl"
    csv_dir = "/content/csv"
    os.makedirs(jsonl_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    if args.finance:
        print("Fetching finance data...")
        data = get_finance_data()
        file_suffix = "finance"
        column_name = "value"
    elif args.energy:
        print(f"Fetching energy data for year: {args.year}...")
        data = get_consumption_data_year(args.year)
        file_suffix = "energy"
        column_name = "consumption"
    elif args.city:
        print(f"Fetching weather data for city: {args.city}...")
        data = get_weather_data(args.city)
        file_suffix = args.city
        column_name = "tavg"
    else:
        print("No valid arguments provided. Please use '--finance', '--energy', or '-city'.")
        return

    # Process the data: split into training (60%) and testing (40%)
    sequence_list = data.tolist() if hasattr(data, "tolist") else list(data)
    split_idx = int(len(sequence_list) * 0.6)
    training_data = sequence_list[:split_idx]
    test_data = sequence_list[split_idx:]

    train_dict = {"sequence": training_data}
    test_dict = {"sequence": test_data}

    with open(f"{jsonl_dir}/training_{file_suffix}.jsonl", "w", encoding="utf-8") as f_train:
        f_train.write(json.dumps(train_dict) + "\n")
    with open(f"{jsonl_dir}/test_{file_suffix}.jsonl", "w", encoding="utf-8") as f_test:
        f_test.write(json.dumps(test_dict) + "\n")

    test_df = pd.DataFrame(test_data, columns=[column_name])
    test_df.to_csv(f"{csv_dir}/test_{file_suffix}.csv", index=False)

if __name__ == "__main__":
    main()
