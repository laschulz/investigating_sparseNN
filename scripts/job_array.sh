#!/bin/bash                      
#SBATCH -t 01:00:00          # walltime = 1 hours and 30 minutes
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --mem=32G                   # Set memory limit (e.g., 16 GB)
#SBATCH --job-name array
#SBATCH --array=0-1            # Running two different students 
#SBATCH --output=/om2/user/laschulz/investigating_sparseNN/logs/output_%A_%a.log
#SBATCH --error=/om2/user/laschulz/investigating_sparseNN/logs/error_%A_%a.log
hostname                     # Print the hostname of the compute node


# Activate the conda environment
source /om2/user/laschulz/anaconda/etc/profile.d/conda.sh
conda activate sparse

# Create the logs directory if it doesn't exist
mkdir -p /om2/user/laschulz/investigating_sparseNN/logs
cd /om2/user/laschulz/investigating_sparseNN

# Define different student_type values
STUDENT_TYPES=("overlappingCNN")

CONFIG_FILES=("config_b16.json" "config_b32.json")  # Add more if needed

# Compute indices
NUM_STUDENTS=${#STUDENT_TYPES[@]}
NUM_CONFIGS=${#CONFIG_FILES[@]}

TOTAL_COMBINATIONS=$((NUM_STUDENTS * NUM_CONFIGS))

# Map SLURM task ID to (student_type, config_file)
STUDENT_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_STUDENTS))
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_STUDENTS))

STUDENT=${STUDENT_TYPES[$STUDENT_INDEX]}
CONFIG=${CONFIG_FILES[$CONFIG_INDEX]}

# Run the Python script with the selected student_type
echo "Running with student_type: $STUDENT"
python src/main.py \
    --mode multiple \
    --student_type $STUDENT
    --config_path $CONFIG