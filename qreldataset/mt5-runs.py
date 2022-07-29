#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/ipykernel/2022a/bin/ipython --ipython-dir=/tmp
#SBATCH --time=0:5:0
#SBATCH --account=def-smucker
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=6GB
#SBATCH --cpus-per-task=4
#SBATCH --array=1-51,101-200,1001-1090
import os

topic = int(os.environ.get('SLURM_ARRAY_TASK_ID', 1))

from torch.cuda import amp
from itertools import chain

from tqdm import tqdm

THRESHOLD = 0.5

try:
    from mt5lib import Query, Text, MonoT5
except:
    from qreldataset.mt5lib import Query, Text, MonoT5
print("Importing...", flush=True)
import argparse
from timeit import default_timer as timer





import pandas as pd

# window, step = 12, 6
# df = pd.read_parquet("qreldataset/2019qrels.passages.parquet")
# df_all = pd.read_parquet(f"data/RunBM25.1k.passages_{window}_{step}/")
df_all = pd.read_parquet(f"data/Top1kBM25.bigbird2_{THRESHOLD*100}_passages.snappy.parquet")
df_all = df_all.rename(columns={"score": "bm25"})

df = df_all[df_all.topic.eq(topic)]
topics = pd.read_csv("./data/topics_fixed_extended.tsv.txt", sep="\t")
# df = df.merge(topics["topic description efficacy".split()], on="topic", how="inner")
df = df.merge(topics["topic efficacy".split()], on="topic", how="inner")

# df = df[df.topic == topic].merge(topics[topics.topic == topic]["topic description efficacy".split()], on="topic", how="inner")
query = Query(topics[topics.topic == topic].iloc[0].description)

texts = [Text(p[1].passage, p[1]) for p in
         df.iterrows()]

reranker = MonoT5(pretrained_model_name_or_path=f"castorini/monot5-base-med-msmarco")

# print("Reranking with MonoT5...", flush=True)
start = timer()
with amp.autocast():
    reranked = reranker.rerank(query, texts)
end = timer()
# print(f"Done. reraking {len(texts)} passages with monot5-{type}-med-msmarco took {end - start} seconds.", flush=True)

top_passage_per_doc = {x.metadata["docno"]: (x, x.score) for x in sorted(reranked, key=lambda x: x.score)}

run = [{"score": x[1], **x[0].metadata.to_dict()} for i, x in enumerate(
    sorted(top_passage_per_doc.values(), key=lambda x: x[1], reverse=True))]

run_df = pd.DataFrame(run)

run_df = run_df.sort_values("topic score".split(), ascending=[True, False])

# run_df.to_parquet(f"data/RunBM25.1k.passages_{window}_{step}.top_mt5/{topic}.snappy.parquet")

Path(f"data/RunBM25.1k.passages_bigbird2_{int(THRESHOLD*100)}.top_mt5").mkdir(parents=True, exist_ok=True)
run_df.to_parquet(f"data/RunBM25.1k.passages_bigbird2_{int(THRESHOLD*100)}.top_mt5/{topic}.snappy.parquet")


#%%
import pandas as pd
from utils.util import fixdocno
dfx = pd.read_parquet(f"data/RunBM25.1k.passages_bigbird2_{int(THRESHOLD*100)}.top_mt5").sort_values("topic score".split(), ascending=[True, False])
dfx["ranking"] = list(range(1,1001)) * dfx.topic.nunique()
run = dfx.apply(lambda x: f"{x.topic} Q0 {fixdocno(x.docno)} {x.ranking} {x.score} WatS-Bigbird2_{int(THRESHOLD*100)}-MT5", axis=1)
run.to_csv(f"runs/WatS-Bigbird2_{int(THRESHOLD*100)}-MT5.all", index=False, header=False)