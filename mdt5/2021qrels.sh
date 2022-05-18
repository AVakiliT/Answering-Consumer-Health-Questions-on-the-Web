#!/usr/bin/env bash
#SBATCH --account=rrg-smucker
#SBATCH --time=0-0:40:0

#SBATCH --array=100-150
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:v100l:1
#SBATCH --output=slurm/%A_%a.out


module load gcc/9.3.0 arrow python scipy-stack
source ~/avakilit/PYTORCH/bin/activate
#pip install --upgrade pip

echo "Starting script..."

ipython mdt5/2021qrels.py -- --topic_no $SLURM_ARRAY_TASK_ID \

