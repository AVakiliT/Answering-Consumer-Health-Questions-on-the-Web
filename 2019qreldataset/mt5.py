import abc
from typing import Optional, List, Mapping, Any

import torch
from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM
# input file - topic
# output
import os

from torch.cuda import amp
from mt5lib import Query, Text, MonoT5
print("Importing...", flush=True)
import argparse
from random import sample
from timeit import default_timer as timer



import pandas as pd
df = pd.read_parquet("2019qreldataset/2019qrels.passages.parquet")

topics = pd.read_csv("./data/topics.csv", sep="\t")

topic = 39

# df = df[df.topic == topic].merge(topics[topics.topic == topic]["topic description efficacy".split()], on="topic", how="inner")
query = Query(topics[topics.topic == topic].iloc[0].description)

texts = [Text(p.passage, {'docid': p.docno, 'url': p.url}, 0) for p in
         df[df.topic == topic].itertuples()]

reranker = MonoT5(pretrained_model_name_or_path=f"castorini/monot5-base-med-msmarco")

print("Reranking with MonoT5...", flush=True)
start = timer()
with amp.autocast():
    reranked = reranker.rerank(query, texts)
end = timer()
print(f"Done. reraking {len(texts)} passages with monot5-{type}-med-msmarco took {end - start} seconds.", flush=True)
