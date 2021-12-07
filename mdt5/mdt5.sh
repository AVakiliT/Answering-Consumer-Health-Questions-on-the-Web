#!/usr/bin/env bash
#SBATCH --account=rrg-smucker
#SBATCH --time=0-12:0:0
#SBATCH --array=101-150
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/slurm-%A_%a.out

#need these build environment
#virtualenv ~/PYGAGGLE
#source ~/PYGAGGLE/bin/activate
#module load rust
#module load swig
#pip install git+https://github.com/castorini/pygaggle.git
#pip install faiss-gpu

module load python
source ~/PYGAGGLE/bin/activate
pip install --upgrade pip
module load java
module load StdEnv/2020  gcc/9.3.0  cuda/11.4
module load faiss
module load arrow

~/PYGAGGLE/bin/python mdt5.py --topic_no $SLURM_ARRAY_TASK_ID \
 --topic_file /project/6004803/smucker/group-data/topics/misinfo-2021-topics.xml \
 --model_type base \
 --bm25run /project/6004803/avakilit/Trec21_Data/Top1kBM25_1p_passages/part-00000-0da9fef6-fd3a-48a8-96d8-f05f4d9e9da2-c000.snappy.parquet

