#!/bin/bash

# Environment setup script for Isaac Lab experiments

# Install Miniconda if not already installed
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    source ~/miniconda3/bin/activate
    conda init --all
else
    echo "Miniconda already installed."
fi

# Create and activate conda environment
conda create --prefix isaaclab_env python=3.10 -y
conda activate ./isaaclab_env

# Install PyTorch
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install Isaac Sim
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# Install additional dependencies
sudo apt-get install libnccl2 libnccl-dev -y
sudo apt install cmake build-essential -y

echo "Environment setup complete."
echo "Please run: conda activate ./isaaclab_env to activate the environment."
