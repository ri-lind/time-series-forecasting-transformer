#!/bin/bash

# Default values
CITY=""
USE_FINANCE=false
USE_GPU=false

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -city)
            if [[ -n "$2" && "$2" != -* ]]; then
                CITY="$2"
                shift 2
            else
                echo "Error: '-city' requires a non-empty argument."
                exit 1
            fi
            ;;
        --finance)
            USE_FINANCE=true
            shift
            ;;
        --gpu)
            USE_GPU=true
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            echo "Usage: bash setup_and_run.sh -city <city_name> [--finance] [--gpu]"
            exit 1
            ;;
    esac
done

# Ensure that at least one data flag is provided
if [ -z "$CITY" ] && [ "$USE_FINANCE" = false ]; then
    echo "Usage: bash setup_and_run.sh -city <city_name> [--finance] [--gpu]"
    exit 1
fi

# Clone the repository
git clone https://github.com/ri-lind/time-series-forecasting-transformer.git
cd time-series-forecasting-transformer || exit

# Checkout the Time-MoE branch
git checkout Time-MoE

# Install virtualenv and create a virtual environment
pip install virtualenv
virtualenv colab_env

# Activate virtual environment
if [ -f "colab_env/bin/activate" ]; then
    source colab_env/bin/activate
else
    echo "Error: Virtual environment activation script not found!"
    exit 1
fi

pip list  # Print installed packages for debugging

# Install dependencies
pip install -r requirements.txt

# Preprocessing steps
# Run finance preprocessing if the flag is set
if [ "$USE_FINANCE" = true ]; then
    echo "Running finance data preprocessing..."
    python3 pre_processing/collect_process.py --finance
fi

# Run weather preprocessing if a city is specified
if [ -n "$CITY" ]; then
    echo "Running weather data preprocessing for city: $CITY..."
    python3 pre_processing/collect_process.py -city "$CITY"
fi

# Training/Model execution
# Run the main script for finance data if applicable
if [ "$USE_FINANCE" = true ]; then
    if [ "$USE_GPU" = true ]; then
        python torch_dist_run.py main.py -d "/content/jsonl/training_finance.jsonl" --save_only_model
    else
        python3 main.py -d "/content/jsonl/training_finance.jsonl" --save_only_model
    fi
fi

# Run the main script for weather data if a city was provided
if [ -n "$CITY" ]; then
    if [ "$USE_GPU" = true ]; then
        python torch_dist_run.py main.py -d "/content/jsonl/training_${CITY}.jsonl" --save_only_model
    else
        python3 main.py -d "/content/jsonl/training_${CITY}.jsonl" --save_only_model
    fi
fi

# Optionally, evaluation command can be run (currently commented out)
# python /content/time-series-forecasting-transformer/run_eval.py -m /content/time-series-forecasting-transformer/logs/time_moe/ -d /content/csv/test_${CITY}.csv --prediction_length 48
