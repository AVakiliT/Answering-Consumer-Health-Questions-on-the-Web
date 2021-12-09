#!/usr/bin/env bash
#SBATCH --account=rrg-smucker
#SBATCH --time=0-12:0:0
#SBATCH --array=1-51
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:v100l:1
#SBATCH --output=slurm/%A_%a.out

#need these build environment
#virtualenv ~/PYGAGGLE
#source ~/PYGAGGLE/bin/activate
#module load rust
#module load swig
#pip install git+https://github.com/castorini/pygaggle.git
#pip install faiss-gpu

module load StdEnv  gcc  cuda/11
module load faiss
echo "Loaded Faiss"
module load arrow
echo "Loaded Arrow"
module load python
source ~/PYGAGGLE/bin/activate
#pip install --upgrade pip
module load java

echo "Starting script..."

#~/PYGAGGLE/bin/python mdt5.py --topic_no $SLURM_ARRAY_TASK_ID \
# --topic_file /project/6004803/smucker/group-data/topics/misinfo-2021-topics.xml \
# --model_type base \
# --bm25run /project/6004803/avakilit/Trec21_Data/Top1kBM25_1p_passages/part-00000-0da9fef6-fd3a-48a8-96d8-f05f4d9e9da2-c000.snappy.parquet

~/PYGAGGLE/bin/python mdt5.py --topic_no $SLURM_ARRAY_TASK_ID \
 --topic_file /project/6004803/avakilit/test/playground/anserini/runs/run.c4.noclean.bm25.topics.2019.txt \
 --model_type base \
 --bm25run /project/6004803/avakilit/Trec21_Data/Top1kBM25_2019_1p_passages/part-00000-a697cfb9-9405-449d-8548-e4ddc6ca9f7a-c000.snappy.parquet

