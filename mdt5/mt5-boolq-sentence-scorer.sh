#!/usr/bin/env bash
#SBATCH --account=rrg-smucker
#SBATCH --time=0-2:0:0

#SBATCH --array=100-150
# --array=1-51
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:v100l:1
#SBATCH --output=slurm/%A_%a.out


module load gcc/9.3.0 arrow python scipy-stack
source ~/avakilit/PYTORCH/bin/activate
#pip install --upgrade pip

echo "Starting script..."

ipython mdt5/mt5-boolq-sentence-scorer.py -- --topic_no $SLURM_ARRAY_TASK_ID \
 --bm25run Top1kBM25_2021

