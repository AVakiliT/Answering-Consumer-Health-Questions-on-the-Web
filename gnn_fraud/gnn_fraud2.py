# %%
import re

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import trange, tqdm
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer
from tldextract import extract
from boolqstuff.bert_modules import BoolQBertModule
import networkx as nx
# df = pd.read_parquet("./mdt5/output_Top1kBM25_2019_mt5_2019_base-med_with_text")
topics = pd.read_csv("./data/topics.csv", sep="\t", index_col="topic")
# v = pd.read_csv("./data/filtered_vertices.tsv", sep="\t", header=None, names="ccid rdomain nhosts domain".split())
# e = pd.read_csv("./data/filtered_edges.tsv", sep="\t", header=None, names="from_ccid to_ccid".split())
v = pd.read_parquet("./data/host-graph/filtered_verticies")
e = pd.read_parquet("./data/host-graph/filtered_edges")
# df["domain"] = df.url.apply(lambda x: extract(x).domain + '.' + extract(x).suffix)
df = pd.read_parquet("./data/Top1kBM25_2021_1p_sentences_with_probs")
df["prob_pos"] = df.probs.apply(lambda x: x[2])
df["prob_neg"] = df.probs.apply(lambda x: x[0])
# df.drop("index")
#fix hosts
df.host = df.host.apply(lambda x: re.sub('^(www)?\.','',x))

# df[df.efficacy != 0].groupby("topic").agg({"pred": "mean", "efficacy":"max", "description":"max"}).sort_values("pred")
# df[df.efficacy.ne(0) & df.domain.isin("nature.com heart.org ca.gov stanford.edu ucla.edu who.int hhs.gov bj.com harvard.edu cdc.gov webmd.com mayoclinic.com nih.gov cochrane.org".split())].groupby("topic").agg({"pred": "mean", "efficacy":"max", "description":"max"}).sort_values("pred")
#%%
df["prob_pos_z"] = (df.prob_pos - df.prob_pos.mean())/df.prob_pos.mean()
df["prob_neg_z"] = (df.prob_neg - df.prob_neg.mean())/df.prob_neg.mean()
#%%
graphs = {}
for i in tqdm(sorted(df.topic.unique().tolist())):
    temp = df[df.topic.eq(i)]
    temp = pd.concat([temp.loc[temp.groupby("docno").prob_neg.idxmax()], temp.loc[temp.groupby("docno").prob_pos.idxmax()]])
    temp = temp[temp.prob_pos.gt(.33) | temp.prob_neg.gt(.33) ]
    filtered_urls = temp.host

    fv = v.merge(filtered_urls, on="host", how="inner")
    fe = e.merge(fv[["id"]].rename(columns={"id":"id_to"}), on="id_to", how="inner")\
        .merge(fv[["id"]].rename(columns={"id":"id_from"}), on="id_from", how="inner")
    G = nx.DiGraph()
    G.add_nodes_from(fv.apply(lambda row: (row.id, dict(host=row.host)), axis=1))
    G.add_edges_from(fe.apply(lambda row: (row.id_from, row.id_to), axis=1))
    graphs[i] = G# [(G.nodes()[k]["host"],v) for k, v in sorted(nx.pagerank(G).items(), key=lambda x: -x[1])][:40]
#%%
topic = 101
temp  = df[
    df.topic.eq(topic)
    ]
temp.loc[temp.groupby("docno").lol.idxmax()].sort_values("lol", ascending=False).passage[:10].to_list()

#%%
temp.loc[temp.groupby("docno").prob_pos.idxmax()].passage[:5].to_list()
for i in temp.loc[temp.groupby("docno").prob_pos.idxmax()].sort_values("prob_pos", ascending=False).passage[:5].to_list():
    print(i)
temp=df[
    df.topic.eq(topic)
    ]
for i in temp.loc[temp.groupby("docno").prob_neg.idxmax()].sort_values("prob_neg", ascending=False).passage[:5].to_list():
    print(i)
print(temp.loc[ temp.groupby("host").prob_neg.idxmax()].sort_values("prob_neg", ascending=False).prob_neg.gt(.33).mean())
print(temp.loc[ temp.groupby("host").prob_pos.idxmax()].sort_values("prob_pos", ascending=False).prob_pos.gt(.33).mean())
#%%
# df["match"] = (df2.prob.gt(.5).astype("long") * 2 - 1).eq(df2.efficacy).astype("float")
df["match"] = df.pred ==
df[df.efficacy != 0].groupby("topic").agg({"match": "mean", "efficacy":"max", "description":"max"}).sort_values("match")

#%%