#!/usr/bin/env bash
#SBATCH --account=rrg-smucker
#SBATCH --time=0-12:0:0
#SBATCH --array=1-4
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

python mdt5.py --topic_no $SLURM_ARRAY_TASK_ID \
 --topic_file /project/6004803/smucker/group-data/topics/misinfo-2021-topics.xml \
 --model_type base \
 --bm25run /project/6004803/avakilit/Trec21_Data/part-00000-2bef8f95-53dc-49f9-8b45-31f5deaf0be1-c000.snappy.parquet

