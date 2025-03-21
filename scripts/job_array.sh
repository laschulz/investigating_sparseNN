#!/bin/bash                      
#SBATCH -t 01:00:00          # walltime = 1 hours and 30 minutes
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --mem=32G                   # Set memory limit (e.g., 16 GB)
#SBATCH --job-name array
#SBATCH --array=0-1            # Running two different students 
hostname                     # Print the hostname of the compute node

# Create the logs directory if it doesn't exist
mkdir -p /om2/user/laschulz/investigating_sparsenn/logs

# Activate the conda environment
source /om2/user/laschulz/anaconda/etc/profile.d/conda.sh
conda activate sparse

cd ..


# Define different student_type values
STUDENT_TYPES=("nonoverlappingViT" "overlappingCNN")

# Get the correct student_type based on SLURM task ID
STUDENT=${STUDENT_TYPES[$SLURM_ARRAY_TASK_ID]}

# Run the Python script with the selected student_type
echo "Running with student_type: $STUDENT"
python src/main.py \
    --mode multiple \
    --student_type $STUDENT