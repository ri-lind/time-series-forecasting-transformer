#!/bin/bash

# Ensure at least one argument (city) is provided
if [ -z "$1" ]; then
    echo "Usage: bash setup_and_run.sh <city_name> [--gpu]"
    exit 1
fi

CITY=$1
USE_GPU=false

# Check if the second argument is --gpu
if [ "$2" == "--gpu" ]; then
    USE_GPU=true
fi

# Clone the repository
git clone https://github.com/ri-lind/time-series-forecasting-transformer.git
cd time-series-forecasting-transformer || exit

# Checkout the Time-MoE branch
git checkout Time-MoE

# Install virtualenv and create a virtual environment
pip install virtualenv
virtualenv colab_env

# Ensure the virtual environment exists before activating
if [ -f "colab_env/bin/activate" ]; then
    source colab_env/bin/activate  # Or use `. colab_env/bin/activate`
else
    echo "Error: Virtual environment activation script not found!"
    exit 1
fi

pip list  # Print installed packages for debugging

# Install dependencies
pip install -r requirements.txt

# Run preprocessing with the city argument (creates the expected directory)
python3 pre_processing/weather.py -city "$CITY"

# Run the main script, with or without GPU
if [ "$USE_GPU" = true ]; then
    python torch_dist_run.py main.py -d "/content/jsonl/${CITY}.jsonl"
else
    python3 main.py -d "/content/jsonl/${CITY}.jsonl"
fi
