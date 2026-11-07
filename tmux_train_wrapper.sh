#!/bin/bash

# Wrapper for training in tmux with proper environment

# Load conda environment if needed
# source ~/.bashrc
# conda activate torchtitan

# Change to project directory
cd /home/sfong7/ITO-HC-llama

# Ensure WANDB_API_KEY is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY not set"
    exit 1
fi

# Run training
exec bash train_identity_tanh_gamma0.001.sh
