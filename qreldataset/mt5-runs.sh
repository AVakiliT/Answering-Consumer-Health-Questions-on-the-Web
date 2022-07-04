#!/usr/bin/env bash
#SBATCH --account=def-smucker
#SBATCH --time=0-1:00:0
#SBATCH --array=0-7
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=slurm/%A_%a.out

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
module load StdEnv gcc cuda/11 faiss arrow/5 python java
source ~/avakilit/PYTORCH/bin/activate

echo "Starting script..."

ipython --ipython-dir=/tmp qreldataset/mt5-runs.py -- --n $SLURM_ARRAY_TASK_ID


