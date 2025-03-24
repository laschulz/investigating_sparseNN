#!/bin/bash                        
#SBATCH -t 01:30:00          # walltime = 1 hours and 30 minutes
#SBATCH -n 1                 # one CPU (hyperthreaded) cores
#SBATCH --gres=gpu:1              # Request 1 GPU  
#SBATCH --output=/om2/user/laschulz/investigating_sparseNN/logs/output_fcnn_%A_%a.log
#SBATCH --error=/om2/user/laschulz/investigating_sparseNN/logs/error_fcnn_%A_%a.log
#SBATCH --job-name fcnn

hostname                     # Print the hostname of the compute node
# Execute commands to run your program here, taking Python for example,

# Create the logs directory if it doesn't exist
mkdir -p /om2/user/laschulz/investigating_sparseNN/logs

# Activate the conda environment
source /om2/user/laschulz/anaconda3/etc/profile.d/conda.sh
conda activate sparse
echo $CONDA_PREFIX

cd /om2/user/laschulz/investigating_sparseNN || exit 1
FILES=$(ls)
echo $FILES
echo $CONDA_PREFIX
which python
python --version

python src/main.py \
    --mode multiple \
    --student_type fcnn_decreasing \
    --config_path config_b16.json