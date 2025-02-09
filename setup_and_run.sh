#!/bin/bash

# Ensure a city argument is provided
if [ -z "$1" ]; then
    echo "Usage: sh setup_and_run.sh <city_name>"
    exit 1
fi

CITY=$1

# Clone the repository
git clone https://github.com/ri-lind/time-series-forecasting-transformer.git
cd time-series-forecasting-transformer || exit

# Checkout the Time-MoE branch
git checkout Time-MoE

# Install virtualenv and create a virtual environment
pip install virtualenv
virtualenv colab_env

# Activate the virtual environment
source colab_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run preprocessing with the city argument
python3 pre_processing/weather.py -city "$CITY"

# Run the main script with the generated JSONL data
python3 main.py -d "/content/jsonl/${CITY}.jsonl"
