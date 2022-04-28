#!/usr/bin/env bash
#SBATCH --account=def-smucker
#SBATCH --time=0-3:0:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:1


module load scipy-stack/2022a gcc arrow cuda/11
source ~/PYTORCH/bin/activate

ipython ./mt5.py