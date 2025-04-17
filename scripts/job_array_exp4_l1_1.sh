#!/bin/bash                      
#SBATCH -t 20:00:00                  # walltime = 1 hour
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=32G                   # Set memory limit
#SBATCH --cpus-per-task=8
#SBATCH --job-name=exp4
#SBATCH --array=0-2                 
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
CONFIG_FILES=("config_l1_1.json")
SEEDS=(2 3 4)

# Compute indices
NUM_CONFIGS=${#CONFIG_FILES[@]}
NUM_SEEDS=${#SEEDS[@]}

CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

CONFIG=${CONFIG_FILES[$CONFIG_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

echo "Running with config: $CONFIG, seed: $SEED"

python src/main.py \
    --mode single \
    --teacher_model baselineCNN_tanh \
    --student_model fcn_128_128_sigmoid \
    --config_path $CONFIG \
    --seed $SEED \
    --name exp4

python src/main.py \
    --mode single \
    --teacher_model baselineCNN_tanh \
    --student_model fcn_128_128_relu \
    --config_path $CONFIG \
    --seed $SEED \
    --name exp4

python src/main.py \
    --mode single \
    --teacher_model baselineCNN_tanh \
    --student_model fcn_128_128_tanh \
    --config_path $CONFIG \
    --seed $SEED \
    --name exp4