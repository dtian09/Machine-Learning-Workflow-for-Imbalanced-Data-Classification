#!/bin/bash
#SBATCH --job-name=dnn_with_joblib
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=dnn_output.log

module load python/3.9
source activate tensorflow_env

python workflow_nn.py
