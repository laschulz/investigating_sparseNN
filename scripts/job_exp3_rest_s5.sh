#!/bin/bash                      
#SBATCH -t 30:00:00                  # walltime = 1 hour
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=32G                   # Set memory limit
#SBATCH --cpus-per-task=8
#SBATCH --job-name=exp3              
#SBATCH --output=/om2/user/laschulz/investigating_sparseNN/logs/output_%A_%a.log
#SBATCH --error=/om2/user/laschulz/investigating_sparseNN/logs/error_%A_%a.log

hostname                             # Print node info

# Activate the conda environment
source /om2/user/laschulz/anaconda3/etc/profile.d/conda.sh
conda activate sparse_new

# Create logs directory if needed
mkdir -p /om2/user/laschulz/investigating_sparseNN/logs

cd /om2/user/laschulz/investigating_sparseNN || exit 1

python src/main.py \
    --mode single \
    --teacher_model baselineCNN_relu \
    --student_model fcn_256_32_relu \
    --config_path config_l2_1.json \
    --seed 1 \
    --name exp3

python src/main.py \
    --mode single \
    --teacher_model baselineCNN_relu \
    --student_model fcn_256_32_relu \
    --config_path config_l2_1.json \
    --seed 4 \
    --name exp3

python src/main.py \
    --mode single \
    --teacher_model baselineCNN_relu \
    --student_model fcn_256_32_relu \
    --config_path config_l2_1.json \
    --seed 2 \
    --name exp3

python src/main.py \
    --mode single \
    --teacher_model baselineCNN_relu \
    --student_model fcn_256_32_sigmoid \
    --config_path config_l2_1.json \
    --seed 2 \
    --name exp3

python src/main.py \
    --mode single \
    --teacher_model baselineCNN_relu \
    --student_model fcn_256_32_relu \
    --config_path config_l2_1.json \
    --seed 3 \
    --name exp3

python src/main.py \
    --mode single \
    --teacher_model baselineCNN_relu \
    --student_model fcn_256_32_sigmoid \
    --config_path config_l2_1.json \
    --seed 3 \
    --name exp3

python src/main.py \
    --mode single \
    --teacher_model baselineCNN_relu \
    --student_model fcn_256_32_relu \
    --config_path config_l2_1.json \
    --seed 5 \
    --name exp3

python src/main.py \
    --mode single \
    --teacher_model baselineCNN_relu \
    --student_model fcn_256_32_sigmoid \
    --config_path config_l2_1.json \
    --seed 5 \
    --name exp3