#!/bin/bash

# Clone the repository if not already present
if [ ! -d "samformer" ]; then
    git clone https://github.com/romilbert/samformer.git
fi

cd samformer || { echo "Failed to enter samformer directory"; exit 1; }

# Install virtualenv (if not already installed) and create a virtual environment named 'colab_env'
pip install virtualenv
virtualenv colab_env

# Activate the virtual environment
if [ -f "colab_env/bin/activate" ]; then
    source colab_env/bin/activate
else
    echo "Error: Virtual environment activation script not found!"
    exit 1
fi

# Upgrade pip and install dependencies from requirements.txt
pip install --upgrade pip
pip install -r requirements.txt