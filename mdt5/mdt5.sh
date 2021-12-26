#!/usr/bin/env bash
#SBATCH --account=rrg-smucker
#SBATCH --time=0-1:0:0
#SBATCH --array=101-150
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/%A_%a.out

#build environment
#virtualenv ~/PYGAGGLE
#source ~/PYGAGGLE/bin/activate
#module load rust
#module load swig
#pip install git+https://github.com/castorini/pygaggle.git

module load StdEnv  gcc  cuda/11
module load faiss
module load arrow
module load python
source ~/PYGAGGLE/bin/activate
#pip install --upgrade pip
module load java

echo "Starting script..."

~/PYGAGGLE/bin/python mdt5.py --topic_no $SLURM_ARRAY_TASK_ID \
 --topic_file /project/6004803/smucker/group-data/topics/misinfo-2021-topics.xml \
 --model_type base-med \
 --no-duo \
 --bm25run Top1kBM25_256l

#~/PYGAGGLE/bin/python mdt5.py --topic_no $SLURM_ARRAY_TASK_ID \
# --topic_file /project/6004803/smucker/group-data/topics/2019topics.xml \
# --model_type base \
# --bm25run /project/6004803/avakilit/Trec21_Data/Top1kBM25_2019_1p_passages/part-00000-a697cfb9-9405-449d-8548-e4ddc6ca9f7a-c000.snappy.parquet

