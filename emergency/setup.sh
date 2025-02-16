#!/bin/bash
set -e

# Clone the repository if not already present
if [ ! -d "samformer" ]; then
    echo "Cloning samformer repository..."
    git clone https://github.com/romilbert/samformer.git
fi

cd samformer || { echo "Failed to enter samformer directory"; exit 1; }

# Install virtualenv (if not already installed) and create a virtual environment named 'colab_env'
echo "Installing virtualenv and creating virtual environment 'colab_env'..."
pip install virtualenv
virtualenv colab_env

# Activate the virtual environment
if [ -f "colab_env/bin/activate" ]; then
    echo "Activating virtual environment..."
    source colab_env/bin/activate
else
    echo "Error: Virtual environment activation script not found!"
    exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages from requirements.txt
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

# Install additional packages (which may not be in requirements.txt) needed by the Python file
echo "Installing additional packages..."
pip install transformers matplotlib numpy sklearn kagglehub

# Run the Python file (which will save plots and results into a folder called 'results')
echo "Running weather_samformer.py..."
python weather_samformer.py
