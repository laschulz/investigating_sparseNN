#!/bin/bash                      
#SBATCH -t 30:00:00                  # walltime = 1 hour
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=32G                   # Set memory limit
#SBATCH --cpus-per-task=8
#SBATCH --job-name=exp5
#SBATCH --array=0-29                 # 2 configs × 5 seeds x 3 Reg = 30 jobs
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
CONFIG_FILES=("config_l1_1.json" "config_l1_0.2.json" "config_l2_1.json" "config_l2_0.2.json" "config_noReg_1.json" "config_noReg_0.2.json")
SEEDS=(1 2 3 4 5)

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
    --teacher_type overlappingCNN \
    --student_type fcnn_decreasing \
    --config_path $CONFIG \
    --seed $SEED \
    --name exp5