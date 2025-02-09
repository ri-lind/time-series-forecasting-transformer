import pandas as pd
from numpy.typing import ArrayLike
import kagglehub
import json
import os
import argparse

def get_weather_data(city: str) -> pd.DataFrame:
    path = kagglehub.dataset_download("gucci1337/weather-of-albania-last-three-years")
    years = [2021, 2022, 2023]
    data_frames = []

    for year in years:
        file_path = f"{path}/data_weather/{city}/{city}{year}.csv"
        df = pd.read_csv(file_path)

        # Ensure there is a `date` column and drop NaN values in `tavg`
        df = df[['date', 'tavg']].dropna(subset=['tavg'])
        data_frames.append(df)

    # Concatenate data for all years
    concatenated_data = pd.concat(data_frames, ignore_index=True)

    return concatenated_data

def main():
    parser = argparse.ArgumentParser(description="Fetch and save weather data for a given city.")
    parser.add_argument("-city", required=True, help="Name of the city")
    args = parser.parse_args()

    city = args.city
    dir_path = "csv"  # Save CSVs in "csv" directory

    # Ensure directory exists
    os.makedirs(dir_path, exist_ok=True)

    # Get full dataset
    weather_data = get_weather_data(city)

    # Split the data into training (60%) and testing (40%)
    split_idx = int(len(weather_data) * 0.6)
    training_data = weather_data.iloc[:split_idx]
    test_data = weather_data.iloc[split_idx:]

    # Save as CSV (ensure it has a `date` column)
    training_data.to_csv(f"{dir_path}/training_{city}.csv", index=False)
    test_data.to_csv(f"{dir_path}/test_{city}.csv", index=False)

if __name__ == "__main__":
    main()
