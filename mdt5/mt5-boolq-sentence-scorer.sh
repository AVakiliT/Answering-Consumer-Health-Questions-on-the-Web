#!/usr/bin/env bash
#SBATCH --account=rrg-smucker
#SBATCH --time=0-2:0:0
#SBATCH --array=1-51
# --array=1-51
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:v100l:1
#SBATCH --output=slurm/%A_%a.out


module load StdEnv  gcc  cuda/11 arrow python
source ~/avakilit/PYTORCH/bin/activate
#pip install --upgrade pip

echo "Starting script..."

python mt5-boolq-sentence-scorer.py --topic_no $SLURM_ARRAY_TASK_ID \
 --bm25run Top1kBM25_2019

