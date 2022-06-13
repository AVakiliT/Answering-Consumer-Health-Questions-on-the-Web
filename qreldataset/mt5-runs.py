import abc
from typing import Optional, List, Mapping, Any

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM
# input file - topic
# output
import os

from torch.cuda import amp
try:
    from mt5lib import Query, Text, MonoT5
except:
    from qreldataset.mt5lib import Query, Text, MonoT5
print("Importing...", flush=True)
import argparse
from random import sample
from timeit import default_timer as timer
parser = argparse.ArgumentParser()
parser.add_argument("--topic", default=101, type=int)
args = parser.parse_known_args()
topic = args[0].topic

import pandas as pd
# df = pd.read_parquet("qreldataset/2019qrels.passages.parquet")
df = pd.read_parquet("data/RunBM25.1k.passages_6_3/")
df = df[df.topic.eq(topic)]
topics = pd.read_csv("./data/RW.txt", sep="\t")
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

top_passage_per_doc = sorted(reranked, key=lambda x: x.score, reverse=True)

run = [{"ranking":i + 1, "score":x.score, **x.metadata.to_dict()} for i, x in enumerate(top_passage_per_doc)]

run_df = pd.DataFrame(run)

run_df.to_parquet(f"data/RunBM25.1k.passages_6_3.mt5/{topic}.snappy.parquet")