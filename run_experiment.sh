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
        # Only run pip install if virtual environment not already set up
        if [ ! -d "colab_env" ]; then
            setup_env
        else
            echo "Virtual environment already exists. Skipping pip install."
        fi
    fi
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

# Preprocessing and training/model execution based on provided argument
if [ "$USE_FINANCE" = true ]; then
    FILE_SUFFIX="finance"
    echo "Running finance data preprocessing..."
    python3 pre_processing/collect_process.py --finance
    if [ "$USE_GPU" = true ]; then
        python torch_dist_run.py main.py -d "/content/jsonl/training_${FILE_SUFFIX}.jsonl" --save_only_model -o /content/time-moe/${FILE_SUFFIX}
    else
        python3 main.py -d "/content/jsonl/training_${FILE_SUFFIX}.jsonl" --save_only_model -o /content/time-moe/${FILE_SUFFIX}
    fi
elif [ "$USE_ENERGY" = true ]; then
    FILE_SUFFIX="energy"
    echo "Running energy data preprocessing for year: $YEAR..."
    python3 pre_processing/collect_process.py --energy --year "$YEAR"
    if [ "$USE_GPU" = true ]; then
        python torch_dist_run.py main.py -d "/content/jsonl/training_${FILE_SUFFIX}.jsonl" --save_only_model -o /content/time-moe/${FILE_SUFFIX}
    else
        python3 main.py -d "/content/jsonl/training_${FILE_SUFFIX}.jsonl" --save_only_model -o /content/time-moe/${FILE_SUFFIX}
    fi
elif [ -n "$USE_HEALTHCARE" ]; then
    FILE_SUFFIX="$USE_HEALTHCARE"
    echo "Running healthcare data preprocessing for country: $USE_HEALTHCARE..."
    python3 pre_processing/collect_process.py -h "$USE_HEALTHCARE"
    if [ "$USE_GPU" = true ]; then
        python torch_dist_run.py main.py -d "/content/jsonl/training_${FILE_SUFFIX}.jsonl" --save_only_model -o /content/time-moe/${FILE_SUFFIX}
    else
        python3 main.py -d "/content/jsonl/training_${FILE_SUFFIX}.jsonl" --save_only_model -o /content/time-moe/${FILE_SUFFIX}
    fi
elif [ -n "$CITY" ]; then
    FILE_SUFFIX="$CITY"
    echo "Running weather data preprocessing for city: $CITY..."
    python3 pre_processing/collect_process.py --city "$CITY"
    if [ "$USE_GPU" = true ]; then
        python torch_dist_run.py main.py -d "/content/jsonl/training_${FILE_SUFFIX}.jsonl" --save_only_model -o /content/time-moe/${FILE_SUFFIX}
    else
        python3 main.py -d "/content/jsonl/training_${FILE_SUFFIX}.jsonl" --save_only_model -o /content/time-moe/${FILE_SUFFIX}
    fi
fi

# Modify config.json to add 128 to horizon_lengths (if it exists)
CONFIG_PATH="/content/time-moe/${FILE_SUFFIX}/config.json"

if [ -f "$CONFIG_PATH" ]; then
    echo "Updating horizon_lengths in $CONFIG_PATH..."
    sed -i '/"horizon_lengths": \[/ s/\]/, 128]/' "$CONFIG_PATH"
else
    echo "Warning: Config file not found at $CONFIG_PATH"
fi



python run_eval.py -m /content/time-moe/${FILE_SUFFIX} -d /content/csv/test_${FILE_SUFFIX}.csv --prediction_length 32 --context_length 64
python run_eval.py -m /content/time-moe/${FILE_SUFFIX} -d /content/csv/test_${FILE_SUFFIX}.csv --prediction_length 64 --context_length 128
python run_eval.py -m /content/time-moe/${FILE_SUFFIX} -d /content/csv/test_${FILE_SUFFIX}.csv --prediction_length 128 --context_length 256

