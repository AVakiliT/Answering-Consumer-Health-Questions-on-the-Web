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
window_size = 6
step = 3
df = pd.read_parquet(f"/project/6004803/avakilit/Trec21_Data/data/qrels.2021.passages_{window_size}_{step}")
# df = df[df.passage.apply(len).gt(0)]
topics = pd.read_csv("./data/topics.csv", sep="\t")
df = df.merge(topics["topic description efficacy".split()], on="topic", how="inner")
runs = []
for topic in tqdm(df.topic.unique().tolist()):

    # df = df[df.topic == topic].merge(topics[topics.topic == topic]["topic description efficacy".split()], on="topic", how="inner")
    query = Query(topics[topics.topic == topic].iloc[0].description)

    texts = [Text(row.passage, row["docno timestamp url usefulness stance credibility passage_index passage description efficacy".split()]) for _, row in
             df[df.topic==topic].iterrows()]

    reranker = MonoT5(pretrained_model_name_or_path=f"castorini/monot5-base-med-msmarco")

    # print("Reranking with MonoT5...", flush=True)
    start = timer()
    with amp.autocast():
        reranked = reranker.rerank(query, texts)
    end = timer()
    # print(f"Done. reraking {len(texts)} passages with monot5-{type}-med-msmarco took {end - start} seconds.", flush=True)
    reranked = sorted(reranked, key=lambda i: i.score)
    run = pd.DataFrame(x.metadata for x in reranked)
    run["mt5"] = [x.score for x in reranked]
    run["passage"] = [x.text for x in reranked]
    top_passage_run = run.loc[run.groupby("docno").mt5.idxmax()]
    runs.append(top_passage_run)

final_df = pd.concat(runs)
final_df.to_parquet(f"./data/qrels.2021.passages_{window_size}_{step}.top_passage_mt5.parquet")