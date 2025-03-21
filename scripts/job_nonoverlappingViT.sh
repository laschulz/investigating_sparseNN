#!/bin/bash                      
#SBATCH -t 01:00:00          # walltime = 1 hours and 30 minutes
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --mem=32G                   # Set memory limit (e.g., 16 GB)
#SBATCH --job-name nonoverlappingViT
#SBATCH --output=/om2/user/laschulz/investigating_sparsenn/logs/output_nonoverlappingViT_%A_%a.log
#SBATCH --error=/om2/user/laschulz/investigating_sparsenn/logs/error_nonoverlappingViT_%A_%a.log

hostname                     # Print the hostname of the compute node

# Create the logs directory if it doesn't exist
mkdir -p /om2/user/laschulz/investigating_sparseNN/logs

# Activate the conda environment
source /om2/user/laschulz/anaconda/etc/profile.d/conda.sh
conda activate sparse

cd ..

python src/main.py \
    --mode multiple \
    --student_type nonoverlappingViT \
    --config_path config_b16.json
