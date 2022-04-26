#!/bin/bash
#SBATCH --array=0-23
#SBATCH --time=1:00:00
#SBATCH--mem-per-cpu=8GB
#SBATCH--cpus-per-task=1
#SBATCH --account=def-smucker

source ~/ENV/bin/activate
module load StdEnv/2020  gcc/9.3.0  cuda/11.4 faiss arrow scipy-stack/2021a

python main-2019-qrel.py "${SLURM_ARRAY_TASK_ID}" 1000