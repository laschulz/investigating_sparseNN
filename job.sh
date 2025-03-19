#!/bin/bash                      
#SBATCH -t 00:30:00          # walltime = 1 hours and 30 minutes
#SBATCH -n 1                 # one CPU (hyperthreaded) cores
hostname                     # Print the hostname of the compute node

python src/main.py \
    --mode multiple \
    --student_type nonoverlappingViT