#!/bin/bash                        
#SBATCH -t 18:00:00        
#SBATCH --gres=gpu:1  # Request exactly 1 GPU
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/om2/user/laschulz/investigating_sparseNN/logs/output_fcnn_%A_%a.log
#SBATCH --error=/om2/user/laschulz/investigating_sparseNN/logs/error_fcnn_%A_%a.log
#SBATCH --job-name AllAct_Weight1-1e-5

hostname                     # Print the hostname of the compute node

# Create the logs directory if it doesn't exist
mkdir -p /om2/user/laschulz/investigating_sparseNN/logs

# Activate the conda environment
source /om2/user/laschulz/anaconda3/etc/profile.d/conda.sh
conda activate sparse_new
echo $CONDA_PREFIX

cd /om2/user/laschulz/investigating_sparseNN || exit 1

python src/main.py \
    --mode single \
    --teacher_model nonoverlapping_CNN_all_tanh \
    --student_model fcnn_decreasing_all_tanh \
    --config_path config_2.json \
    --seed 1

python src/main.py \
    --mode single \
    --teacher_model nonoverlapping_CNN_all_tanh \
    --student_model fcnn_decreasing_all_tanh \
    --config_path config_2.json \
    --seed 2

python src/main.py \
    --mode single \
    --teacher_model nonoverlapping_CNN_all_tanh \
    --student_model fcnn_decreasing_all_tanh \
    --config_path config_2.json \
    --seed 4

python src/main.py \
    --mode single \
    --teacher_model nonoverlapping_CNN_all_tanh \
    --student_model fcnn_decreasing_all_tanh \
    --config_path config_2.json \
    --seed 5