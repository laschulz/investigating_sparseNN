#!/bin/bash                      
#SBATCH -t 30:00:00                  # walltime = 1 hour
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=32G                   # Set memory limit
#SBATCH --cpus-per-task=8
#SBATCH --job-name=exp4
#SBATCH --array=0-5                 # 2 configs × 3 seeds x 1 Reg = 6 jobs
#SBATCH --output=/om2/user/laschulz/investigating_sparseNN/logs/output_%A_%a.log
#SBATCH --error=/om2/user/laschulz/investigating_sparseNN/logs/error_%A_%a.log

hostname                             # Print node info

# Activate the conda environment
source /om2/user/laschulz/anaconda3/etc/profile.d/conda.sh
conda activate sparse_new

# Create logs directory if needed
mkdir -p /om2/user/laschulz/investigating_sparseNN/logs

cd /om2/user/laschulz/investigating_sparseNN || exit 1

# Define configs and student model
CONFIG_FILES=("config_noReg_1.json" "config_noReg_0.2.json")
SEEDS=(1 3 5)

# Compute indices
NUM_CONFIGS=${#CONFIG_FILES[@]}
NUM_SEEDS=${#SEEDS[@]}

CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

CONFIG=${CONFIG_FILES[$CONFIG_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

echo "Running with config: $CONFIG, seed: $SEED"

python src/main.py \
    --mode multiple \
    --teacher_type baselineCNN \
    --student_type fcn_128_128 \
    --config_path $CONFIG \
    --seed $SEED \
    --name exp4