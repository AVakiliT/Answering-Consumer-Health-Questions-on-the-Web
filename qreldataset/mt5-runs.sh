#!/usr/bin/env bash
#SBATCH --account=rrg-smucker
#SBATCH --time=0-0:30:0
#SBATCH --array=1001-1090
# --array=1-51
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:v100l:1
#SBATCH --output=slurm/%A_%a.out


module load StdEnv gcc cuda/11 faiss arrow/5 python java
source ~/avakilit/PYTORCH/bin/activate

echo "Starting script..."

ipython --ipython-dir=/tmp qreldataset/mt5-runs.py -- --topic $SLURM_ARRAY_TASK_ID


