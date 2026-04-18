#!/usr/bin/env zsh
#SBATCH --job-name=ocpq-lora
#SBATCH --output=ocpq-lora-%J.out
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

# Load modules (adjust to RWTH specific Python/CUDA modules)
module load python/3.11 cuda/12.1

# Activate your environment
source ~/miniconda3/bin/activate apm_env

# Install Unsloth (if you haven't yet)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

# Run script
python train_unsloth.py
