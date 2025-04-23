#!/bin/bash                      
#SBATCH -t 15:00:00                  # walltime = 1 hour
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=16G                   # Set memory limit
#SBATCH --cpus-per-task=8
#SBATCH --job-name=array
#SBATCH --array=0-4
#SBATCH --output=/om2/user/laschulz/investigating_sparseNN/logs/output_%A_%a.log
#SBATCH --error=/om2/user/laschulz/investigating_sparseNN/logs/error_%A_%a.log

hostname                             # Print node info

# Activate the conda environment
source /om2/user/laschulz/anaconda3/etc/profile.d/conda.sh
conda activate sparse_new

# Create logs directory if needed
mkdir -p /om2/user/laschulz/investigating_sparseNN/logs

cd /om2/user/laschulz/investigating_sparseNN || exit 1

SEEDS=(42)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "Running with seed: $SEED"

python src/main.py \
    --mode multiple \
    --teacher_type baselineCNN \
    --student_type baselineCNN \
    --config_path config.json \
    --seed $SEED