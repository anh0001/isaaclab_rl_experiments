#!/bin/bash

# Environment setup script for Isaac Lab experiments

# Install Miniconda if not already installed
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    eval "$(~/miniconda3/bin/conda shell.bash hook)"
    conda init bash
else
    echo "Miniconda already installed."
fi

# Create conda environment
echo "Creating isaaclab environment..."
conda create --prefix ./isaaclab_env python=3.10 -y

# Source conda to ensure we can activate environments
source ~/.bashrc  # Or use: eval "$(conda shell.bash hook)"

# Activate the environment
echo "Activating environment..."
conda activate ./isaaclab_env

# Install PyTorch with specific version required by isaaclab
echo "Installing PyTorch 2.5.1..."
conda install pytorch=2.5.1 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install Isaac Sim
echo "Installing Isaac Sim..."
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# Install additional dependencies
echo "Installing system dependencies..."
sudo apt-get install libnccl2 libnccl-dev -y
sudo apt install cmake build-essential -y

echo "Environment setup complete."
echo "Please run: conda activate ./isaaclab_env to activate the environment."