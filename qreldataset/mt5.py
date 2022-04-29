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



import pandas as pd
df = pd.read_parquet("qreldataset/2019qrels.passages.parquet")
df = df[df.text.apply(len).gt(0)]
topics = pd.read_csv("./data/topics.csv", sep="\t")
df = df.merge(topics["topic description efficacy".split()], on="topic", how="inner")
runs = []
for topic in tqdm(df.topic.unique().tolist()):

    # df = df[df.topic == topic].merge(topics[topics.topic == topic]["topic description efficacy".split()], on="topic", how="inner")
    query = Query(topics[topics.topic == topic].iloc[0].description)

    texts = [Text(p.passage, {"metadata": (*p[1:6],p[7],*p[9:])}, 0) for p in
             df[df.topic==topic].itertuples()]

    reranker = MonoT5(pretrained_model_name_or_path=f"castorini/monot5-base-med-msmarco")

    # print("Reranking with MonoT5...", flush=True)
    start = timer()
    with amp.autocast():
        reranked = reranker.rerank(query, texts)
    end = timer()
    # print(f"Done. reraking {len(texts)} passages with monot5-{type}-med-msmarco took {end - start} seconds.", flush=True)

    top_passage_per_doc = sorted(list(
        {x.metadata['metadata']: x for x in sorted(reranked, key=lambda i: i.score)}
            .values()),
        key=lambda i: i.score, reverse=True)

    run = [(topic, 0,  i + 1, x.score, *x.metadata["metadata"][1:], x.text) for i, x in enumerate(top_passage_per_doc)]

    run_df = pd.DataFrame(run, columns="topic iter rank score docno usefulness credibility stance url description efficacy text".split())
    runs.append(run_df)

final_df = pd.concat(runs)
final_df.to_parquet("./qreldataset/2019_mt5_dataset.parquet")