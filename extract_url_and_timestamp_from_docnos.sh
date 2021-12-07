#!/bin/bash
#SBATCH --account=rrg-smucker
#SBATCH --time=0-1:0:0
#SBATCH --array=0-55
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --output=slurm/slurm-%A_%a.out


module load gcc arrow python scipy-stack
python extract_url_and_timestamp_from_docnos.py $SLURM_ARRAY_TASK_ID 128
