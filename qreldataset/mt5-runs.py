# input file - topic
# output

from torch.cuda import amp
from itertools import chain

from tqdm import tqdm

try:
    from mt5lib import Query, Text, MonoT5
except:
    from qreldataset.mt5lib import Query, Text, MonoT5
print("Importing...", flush=True)
import argparse
from timeit import default_timer as timer

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int)
parser.add_argument("--k", type=int, default=25)
args = parser.parse_known_args()
k = args[0].k
n = args[0].n
# topic = 1001
topics = list(chain(range(1, 51 + 1), range(101, 150 + 1), range(1001, 1090 + 1)))
topics_subset = topics[n * k: n * k + k]
import pandas as pd

window, step = 12, 6
# df = pd.read_parquet("qreldataset/2019qrels.passages.parquet")
df_all = pd.read_parquet(f"data/RunBM25.1k.passages_{window}_{step}/")
for topic in tqdm():
    df = df_all[df_all.topic.eq(topic)]
    topics = pd.read_csv("./data/topics_fixed_extended.tsv.txt", sep="\t")
    df = df.merge(topics["topic description efficacy".split()], on="topic", how="inner")

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

    run = [{"ranking": i + 1, "score": x[1], **x[0].metadata.to_dict()} for i, x in enumerate(
        sorted(top_passage_per_doc.values(), key=lambda x: x[1], reverse=True))]

    run_df = pd.DataFrame(run)

    run_df.to_parquet(f"data/RunBM25.1k.passages_{window}_{step}.top_mt5/{topic}.snappy.parquet")