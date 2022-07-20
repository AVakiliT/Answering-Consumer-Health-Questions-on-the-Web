#!/bin/bash
#SBATCH --account=rrg-smucker
#SBATCH --time=0-1:0:0
#SBATCH --array=0-55
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --output=slurm/slurm-%A_%a.out


#module load gcc arrow python scipy-stack
#python gettop1kdocs.py $SLURM_ARRAY_TASK_ID 128 Top1kBM25 \
#  /project/6004803/smucker/group-data/runs/trec2021-misinfo/automatic/run.c4.noclean.bm25.topics.2021.10K.fixed_docno.txt

module load gcc arrow python scipy-stack
python gettop1kdocs.py $SLURM_ARRAY_TASK_ID 128 Top1k_2022_BM25 \
  /project/6004803/smucker/group-data/tools/anserini/runs/run.c4noclean.bm25.topics.2022.hits.1k.txt

#python gettop1kdocs.py $SLURM_ARRAY_TASK_ID 128 Top1kBM25_2019 \
#/project/6004803/avakilit/test/playground/anserini/runs/run.c4.noclean.bm25.topics.2019.txt
