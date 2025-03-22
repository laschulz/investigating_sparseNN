#!/bin/bash    

#SBATCH --gres=gpu:1              # Request 1 GPU                  
#SBATCH -t 01:00:00        
#SBATCH --mem=64G                   
#SBATCH --job-name fcnn_decreasing
#SBATCH --output=/om2/user/laschulz/investigating_sparseNN/logs/output_nonoverlappingViT_%A_%a.log
#SBATCH --error=/om2/user/laschulz/investigating_sparseNN/logs/error_nonoverlappingViT_%A_%a.log

hostname                     # Print the hostname of the compute node

# Create the logs directory if it doesn't exist
mkdir -p /om2/user/laschulz/investigating_sparseNN/logs

# Activate the conda environment
source /om2/user/laschulz/anaconda3/etc/profile.d/conda.sh
conda activate sparse
echo $CONDA_PREFIX

cd /om2/user/laschulz/investigating_sparseNN

python src/main.py \
    --mode multiple \
    --student_type fcnn_decreasing \
    --config_path config_b16.json
