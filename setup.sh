#!/bin/bash

# Default values
CITY=""
USE_FINANCE=false
USE_ENERGY=false
USE_HEALTHCARE=""
USE_GPU=false
YEAR=2024
FILE_SUFFIX=""
SETUP_ONLY=false

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -c|--city)
            if [[ -n "$2" && "$2" != -* ]]; then
                CITY="$2"
                shift 2
            else
                echo "Error: '-c|--city' requires a non-empty argument."
                exit 1
            fi
            ;;
        -f|--finance)
            USE_FINANCE=true
            shift
            ;;
        -e|--energy)
            USE_ENERGY=true
            shift
            ;;
        -h|--healthcare)
            if [[ -n "$2" && "$2" != -* ]]; then
                USE_HEALTHCARE="$2"
                shift 2
            else
                echo "Error: '-h|--healthcare' requires a non-empty argument."
                exit 1
            fi
            ;;
        --year)
            if [[ -n "$2" && "$2" != -* ]]; then
                YEAR="$2"
                shift 2
            else
                echo "Error: '--year' requires a valid year argument."
                exit 1
            fi
            ;;
        --gpu)
            USE_GPU=true
            shift
            ;;
        --setup-only)
            SETUP_ONLY=true
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            echo "Usage: bash setup_and_run.sh [--setup-only] (-c <city_name> | -f | -e [--year <year>] | -h <Countryname>) [--gpu]"
            exit 1
            ;;
    esac
done

# If setup-only flag is passed, only perform virtual environment creation and dependency installation.
if [ "$SETUP_ONLY" = true ]; then
    echo "Running setup-only: Creating virtual environment and installing dependencies..."
    if [ ! -d "time-series-forecasting-transformer" ]; then
        git clone https://github.com/ri-lind/time-series-forecasting-transformer.git
    fi
    cd time-series-forecasting-transformer || exit
    git checkout Time-MoE

    pip install virtualenv
    virtualenv colab_env

    if [ -f "colab_env/bin/activate" ]; then
        source colab_env/bin/activate
    else
        echo "Error: Virtual environment activation script not found!"
        exit 1
    fi

    pip list  # Print installed packages for debugging
    pip install -r requirements.txt

    echo "Setup completed successfully."
    exit 0
fi

# Ensure exactly one data flag is provided (only if not running setup-only)
flag_count=0
if [ -n "$CITY" ]; then flag_count=$((flag_count+1)); fi
if [ "$USE_FINANCE" = true ]; then flag_count=$((flag_count+1)); fi
if [ "$USE_ENERGY" = true ]; then flag_count=$((flag_count+1)); fi
if [ -n "$USE_HEALTHCARE" ]; then flag_count=$((flag_count+1)); fi

if [ "$flag_count" -ne 1 ]; then
    echo "Error: Specify exactly one data type: -c <city_name>, -f, -e [--year <year>], or -h <Countryname>."
    echo "Usage: bash setup_and_run.sh (-c <city_name> | -f | -e [--year <year>] | -h <Countryname>) [--gpu]"
    exit 1
fi

# Clone the repository if not already present
if [ ! -d "time-series-forecasting-transformer" ]; then
    git clone https://github.com/ri-lind/time-series-forecasting-transformer.git
fi
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
pip install -r requirements.txt

# Preprocessing and training/model execution based on provided argument
if [ "$USE_FINANCE" = true ]; then
    FILE_SUFFIX="finance"
    echo "Running finance data preprocessing..."
    python3 pre_processing/collect_process.py --finance
    if [ "$USE_GPU" = true ]; then
        python torch_dist_run.py main.py -d "/content/jsonl/training_${FILE_SUFFIX}.jsonl" --save_only_model
    else
        python3 main.py -d "/content/jsonl/training_${FILE_SUFFIX}.jsonl" --save_only_model
    fi
elif [ "$USE_ENERGY" = true ]; then
    FILE_SUFFIX="energy"
    echo "Running energy data preprocessing for year: $YEAR..."
    python3 pre_processing/collect_process.py --energy --year "$YEAR"
    if [ "$USE_GPU" = true ]; then
        python torch_dist_run.py main.py -d "/content/jsonl/training_${FILE_SUFFIX}.jsonl" --save_only_model
    else
        python3 main.py -d "/content/jsonl/training_${FILE_SUFFIX}.jsonl" --save_only_model
    fi
elif [ -n "$USE_HEALTHCARE" ]; then
    FILE_SUFFIX="$USE_HEALTHCARE"
    echo "Running healthcare data preprocessing for country: $USE_HEALTHCARE..."
    python3 pre_processing/collect_process.py -h "$USE_HEALTHCARE"
    if [ "$USE_GPU" = true ]; then
        python torch_dist_run.py main.py -d "/content/jsonl/training_${FILE_SUFFIX}.jsonl" --save_only_model
    else
        python3 main.py -d "/content/jsonl/training_${FILE_SUFFIX}.jsonl" --save_only_model
    fi
elif [ -n "$CITY" ]; then
    FILE_SUFFIX="$CITY"
    echo "Running weather data preprocessing for city: $CITY..."
    python3 pre_processing/collect_process.py --city "$CITY"
    if [ "$USE_GPU" = true ]; then
        python torch_dist_run.py main.py -d "/content/jsonl/training_${FILE_SUFFIX}.jsonl" --save_only_model
    else
        python3 main.py -d "/content/jsonl/training_${FILE_SUFFIX}.jsonl" --save_only_model
    fi
fi

# Optionally, evaluation command using the proper FILE_SUFFIX
# python run_eval.py -m logs/time_moe/ -d csv/test_${FILE_SUFFIX}.csv --prediction_length 64 --context_length 128
