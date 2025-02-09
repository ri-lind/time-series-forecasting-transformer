import pandas as pd
from numpy.typing import ArrayLike
import kagglehub
import json
import os

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


city = "sample_city"
dir_path = "/content/jsonl"

# Ensure the directory exists
os.makedirs(dir_path, exist_ok=True)
avg_temperatures = get_weather_data(city)

# Suppose avg_temperatures is a 1D NumPy array or a Pandas Series.
# Convert it to a list
sequence_list = avg_temperatures.tolist()

data_dict = {"sequence": sequence_list}

# Write to a .jsonl file (with a single line)
with open(f"{dir_path}/{city}.jsonl", "w", encoding="utf-8") as f_out:
    json_line = json.dumps(data_dict)
    f_out.write(json_line + "\n")