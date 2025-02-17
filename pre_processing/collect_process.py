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
from sklearn.preprocessing import MinMaxScaler  # Import the scaler

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
    
    # Apply MinMaxScaler
    scaler = MinMaxScaler()
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
    
    # Apply MinMaxScaler
    scaler = MinMaxScaler()
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
    
    # Apply MinMaxScaler
    scaler = MinMaxScaler()
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
    
    # Apply MinMaxScaler
    scaler = MinMaxScaler()
    scaled_cases = scaler.fit_transform(sliced_cases.reshape(-1, 1)).flatten()
    
    return scaled_cases

def main():
    # Disable default help (-h/--help) so that -h can be used for healthcare data.
    parser = argparse.ArgumentParser(
        description="Fetch and save weather, finance, energy, or healthcare data based on provided arguments.",
        add_help=False
    )
    # Custom help only via --help.
    parser.add_argument('--help', action='help', help='Show this help message and exit.')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--finance", action="store_true", help="Fetch finance data")
    group.add_argument("-e", "--energy", action="store_true", help="Fetch energy data")
    group.add_argument("-c", "--city", help="Name of the city for weather data")
    # Now -h is used for healthcare data.
    group.add_argument("-h", "--healthcare", help="Name of the country for healthcare data")
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
    elif args.healthcare:
        print(f"Fetching healthcare data for country: {args.healthcare}...")
        data = get_healthcare_data(args.healthcare)
        file_suffix = args.healthcare
        column_name = "new_cases"
    else:
        print("No valid arguments provided. Please use one of: --finance, --energy, --city, or --healthcare.")
        return

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
