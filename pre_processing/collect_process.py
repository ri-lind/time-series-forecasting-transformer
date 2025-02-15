import pandas as pd
from numpy.typing import ArrayLike
import kagglehub
import json
import os
import argparse

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
    Returns a numpy array containing the column with the actual data
    """
    CSV_FILE_ABSOLUTE_PATH = "/content/AMZN-stock-price.csv"

    df = pd.read_csv(CSV_FILE_ABSOLUTE_PATH)

    df = df.iloc[:, 1]

    return df.values

def main():
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(
        description="Fetch and save weather or finance data based on provided arguments."
    )
    parser.add_argument("--finance", action="store_true", help="Fetch finance data")
    parser.add_argument("-city", help="Name of the city for weather data")
    args = parser.parse_args()

    jsonl_dir = "/content/jsonl"
    csv_dir = "/content/csv"

    # Ensure the directories exist
    os.makedirs(jsonl_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # If the finance flag is passed, run get_finance_data
    if args.finance:
        print("Fetching finance data...")
        finance_data = get_finance_data()
        # Convert data to list (adjust if your data type is different)
        data_list = finance_data.tolist() if hasattr(finance_data, "tolist") else list(finance_data)

        # Split the data (60% training, 40% testing)
        split_idx = int(len(data_list) * 0.6)
        training_data = data_list[:split_idx]
        test_data = data_list[split_idx:]

        # Create dictionaries for JSONL
        train_dict = {"sequence": training_data}
        test_dict = {"sequence": test_data}

        # Write training data to JSONL
        with open(f"{jsonl_dir}/training_finance.jsonl", "w", encoding="utf-8") as f_train:
            f_train.write(json.dumps(train_dict) + "\n")

        # Write test data to JSONL
        with open(f"{jsonl_dir}/test_finance.jsonl", "w", encoding="utf-8") as f_test:
            f_test.write(json.dumps(test_dict) + "\n")

        # Save test data as a CSV file with one column
        test_df = pd.DataFrame(test_data, columns=["value"])
        test_df.to_csv(f"{csv_dir}/test_finance.csv", index=False)
    
    # Else if a city is provided, run get_weather_data
    elif args.city:
        print(f"Fetching weather data for city: {args.city}...")
        avg_temperatures = get_weather_data(args.city)
        # Convert data to list (if it is a NumPy array or Pandas Series)
        sequence_list = avg_temperatures.tolist() if hasattr(avg_temperatures, "tolist") else list(avg_temperatures)

        # Split the data (60% training, 40% testing)
        split_idx = int(len(sequence_list) * 0.6)
        training_data = sequence_list[:split_idx]
        test_data = sequence_list[split_idx:]

        # Create dictionaries for JSONL
        train_dict = {"sequence": training_data}
        test_dict = {"sequence": test_data}

        # Write training data to JSONL
        with open(f"{jsonl_dir}/training_{args.city}.jsonl", "w", encoding="utf-8") as f_train:
            f_train.write(json.dumps(train_dict) + "\n")

        # Write test data to JSONL
        with open(f"{jsonl_dir}/test_{args.city}.jsonl", "w", encoding="utf-8") as f_test:
            f_test.write(json.dumps(test_dict) + "\n")

        # Save test data as a CSV file with one column
        test_df = pd.DataFrame(test_data, columns=["tavg"])
        test_df.to_csv(f"{csv_dir}/test_{args.city}.csv", index=False)
    
    # If neither finance nor city arguments are provided, print a message
    else:
        print("No valid arguments provided. Please use '--finance' to fetch finance data or '-city' to fetch weather data.")

if __name__ == "__main__":
    main()