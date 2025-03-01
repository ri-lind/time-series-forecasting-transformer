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

REPO_DIR="time-series-forecasting-transformer"
GIT_URL="https://github.com/ri-lind/time-series-forecasting-transformer.git"
BRANCH="Time-MoE"

# Function to set up the virtual environment and install dependencies
setup_env() {
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
}

# Setup-only mode: run only environment setup if needed
if [ "$SETUP_ONLY" = true ]; then
    echo "Running setup-only: Creating virtual environment and installing dependencies..."
    if [ ! -d "$REPO_DIR" ]; then
        git clone "$GIT_URL"
        cd "$REPO_DIR" || exit
        git checkout "$BRANCH"
        setup_env
    else
        cd "$REPO_DIR" || exit
        if [ ! -d "colab_env" ]; then
            setup_env
        else
            echo "Virtual environment already exists. Skipping pip install."
        fi
    fi
    echo "Setup completed successfully."
    exit 0
fi

# Ensure exactly one data flag is provided
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

# Clone the repository if not already present; otherwise, use the existing one.
if [ ! -d "$REPO_DIR" ]; then
    git clone "$GIT_URL"
    cd "$REPO_DIR" || exit
    git checkout "$BRANCH"
    setup_env
else
    cd "$REPO_DIR" || exit
    echo "Repository directory exists. Skipping pip install."
fi

# Set the model directory variable.
MODEL_DIR="/content/downloaded-time-moe"
# Download the HF repo and modify configuration.
python3 setup_timemoe_model.py --absolute_model_location "$MODEL_DIR"

# Data preprocessing based on provided argument.
if [ "$USE_FINANCE" = true ]; then
    FILE_SUFFIX="finance"
    echo "Running finance data preprocessing..."
    python3 pre_processing/collect_process.py --finance
elif [ "$USE_ENERGY" = true ]; then
    FILE_SUFFIX="energy"
    echo "Running energy data preprocessing for year: $YEAR..."
    python3 pre_processing/collect_process.py --energy --year "$YEAR"
elif [ -n "$USE_HEALTHCARE" ]; then
    FILE_SUFFIX="$USE_HEALTHCARE"
    echo "Running healthcare data preprocessing for country: $USE_HEALTHCARE..."
    python3 pre_processing/collect_process.py -h "$USE_HEALTHCARE"
elif [ -n "$CITY" ]; then
    FILE_SUFFIX="$CITY"
    echo "Running weather data preprocessing for city: $CITY..."
    python3 pre_processing/collect_process.py --city "$CITY"
fi

# Set output directory for model saving; this can be reused in subsequent calls.
OUTPUT_DIR="/content/time-moe/${FILE_SUFFIX}"

# Call main.py using the same command irrespective of data flag.
if [ "$USE_GPU" = true ]; then
    python torch_dist_run.py main.py -d "/content/jsonl/training_${FILE_SUFFIX}.jsonl" --save_only_model -o "$OUTPUT_DIR" -m "$MODEL_DIR"
else
    python3 main.py -d "/content/jsonl/training_${FILE_SUFFIX}.jsonl" --save_only_model -o "$OUTPUT_DIR" -m "$MODEL_DIR"
fi

# Run evaluation for different context/prediction lengths.
python run_eval.py -m "$OUTPUT_DIR" -d /content/csv/test_${FILE_SUFFIX}.csv --prediction_length 32 --context_length 64
python run_eval.py -m "$OUTPUT_DIR" -d /content/csv/test_${FILE_SUFFIX}.csv --prediction_length 64 --context_length 128
python run_eval.py -m "$OUTPUT_DIR" -d /content/csv/test_${FILE_SUFFIX}.csv --prediction_length 128 --context_length 256
