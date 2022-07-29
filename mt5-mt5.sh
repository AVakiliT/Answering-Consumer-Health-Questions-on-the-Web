#!/usr/bin/env bash
#SBATCH --time=0:35:0
#SBATCH --account=rrg-smucker
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem-per-cpu=6GB
#SBATCH --cpus-per-task=4
#SBATCH --array=1-51,101-200,1001-1090

module load scipy-stack gcc/9.3 arrow/8 java cuda/11;source ~/avakilit/PYTORCH/bin/activate;ipython --ipython-dir=/tmp mt5-mt5.py  $SLURM_ARRAY_TASK_ID
