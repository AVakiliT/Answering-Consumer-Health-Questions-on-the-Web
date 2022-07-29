#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/ipykernel/2022a/bin/ipython --ipython-dir=/tmp
# SBATCH --time=3:0:0
#SBATCH --time=0:55:0
#SBATCH --account=rrg-smucker
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem-per-cpu=6GB
#SBATCH --cpus-per-task=4
# SBATCH --array=1-51,101-200,1001-1090
#SBATCH --array=117
import os

from utils.util import shell_cmd

shell_cmd('module load StdEnv gcc cuda/11 faiss arrow/8 java')
import sys

import torch
from qreldataset.mt5lib import MonoT5, Query, Text
import pandas as pd
from tqdm import tqdm, trange

topic = int(os.environ.get('SLURM_ARRAY_TASK_ID', 101))

reranker = MonoT5(pretrained_model_name_or_path=f"castorini/monot5-base-med-msmarco")
reranker.model.eval()
df = pd.read_parquet(
    # f"data/RunBM25.1k.passages_6_3_t/topic_{topic}.snappy.parquet"
    f"qreldataset/Qrels.2021.passages_6_3_t/topic_{topic}.snappy.parquet"
)
topics = pd.read_csv("./data/topics_fixed_extended.tsv.txt", sep="\t")
# df = df.merge(topics["topic description efficacy".split()], on="topic", how="inner")
df = df.merge(topics["topic efficacy".split()], on="topic", how="inner")

# df = df[df.topic == topic].merge(topics[topics.topic == topic]["topic description efficacy".split()], on="topic", how="inner")
query = Query(topics[topics.topic == topic].iloc[0].description)

texts = [Text(p[1].passage, p[1]) for p in
         df.iterrows()]
with torch.no_grad():
    reranked = reranker.rerank(query, texts)

top_passage_per_doc = {x.metadata["docno"]: (x, x.score) for x in sorted(reranked, key=lambda x: x.score)}

run = [{"score": x[1], **x[0].metadata.to_dict()} for i, x in enumerate(
    sorted(top_passage_per_doc.values(), key=lambda x: x[1], reverse=True))]

run_df = pd.DataFrame(run)

run_df = run_df.sort_values("topic score".split(), ascending=[True, False])
run_df.to_parquet(
    # f"data/RunBM25.1k.passages_mt5.top_mt5/{topic}.snappy.parquet"
    f"qreldataset/Qrels.2021.passages_6_3.top_mt5/topic_{topic}.snappy.parquet"
)
