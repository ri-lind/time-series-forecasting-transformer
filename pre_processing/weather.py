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

def main():
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(description="Fetch and save weather data for a given city.")
    parser.add_argument("-city", required=True, help="Name of the city")
    args = parser.parse_args()

    city = args.city
    dir_path = "/content/jsonl"
    csv_path = "/content/csv"

    # Ensure the directories exist
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(csv_path, exist_ok=True)

    avg_temperatures = get_weather_data(city)

    # Convert NumPy array/Pandas Series to a list
    sequence_list = avg_temperatures.tolist()

    # Split the data (60% training, 40% testing)
    split_idx = int(len(sequence_list) * 0.6)
    training_data = sequence_list[:split_idx]
    test_data = sequence_list[split_idx:]

    # Create dictionaries for JSONL
    train_dict = {"sequence": training_data}
    test_dict = {"sequence": test_data}

    # Write training data to JSONL
    with open(f"{dir_path}/training_{city}.jsonl", "w", encoding="utf-8") as f_train:
        f_train.write(json.dumps(train_dict) + "\n")

    # Write test data to JSONL
    with open(f"{dir_path}/test_{city}.jsonl", "w", encoding="utf-8") as f_test:
        f_test.write(json.dumps(test_dict) + "\n")

    # Save test data as a CSV file with one column
    test_df = pd.DataFrame(test_data, columns=["tavg"])
    test_df.to_csv(f"{csv_path}/test_{city}.csv", index=False)

if __name__ == "__main__":
    main()
