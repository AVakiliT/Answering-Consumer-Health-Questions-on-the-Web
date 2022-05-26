#!/usr/bin/env bash
#SBATCH --account=rrg-smucker
#SBATCH --time=0-3:40:0


#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:v100l:1


module load gcc/9.3.0 arrow python scipy-stack
source ~/avakilit/PYTORCH/bin/activate
#pip install --upgrade pip

echo "Starting script..."

ipython --ipython-dir=/tmp mdt5/2021qrels_mt5.py

