# %%
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import trange, tqdm
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer
from tldextract import extract
from boolq.bert_modules import BoolQBertModule
import networkx as nx
# df = pd.read_parquet("./mdt5/output_Top1kBM25_2019_mt5_2019_base-med_with_text")
topics = pd.read_csv("./data/topics.csv", sep="\t", index_col="topic")
v = pd.read_csv("./data/filtered_vertices.tsv", sep="\t", header=None, names="ccid rdomain nhosts domain".split())
e = pd.read_csv("./data/filtered_edges.tsv", sep="\t", header=None, names="from_ccid to_ccid".split())
# df["domain"] = df.url.apply(lambda x: extract(x).domain + '.' + extract(x).suffix)
df = pd.read_parquet("./gnn_fraud/temp_df_aug")
df2 = pd.read_parquet("./gnn_fraud/temp_df")
df["pred"] = df.probs.apply(lambda x: x.argmax())
df[df.efficacy != 0].groupby("topic").agg({"pred": "mean", "efficacy":"max", "description":"max"}).sort_values("pred")
df[df.efficacy.ne(0) & df.domain.isin("nature.com heart.org ca.gov stanford.edu ucla.edu who.int hhs.gov bj.com harvard.edu cdc.gov webmd.com mayoclinic.com nih.gov cochrane.org".split())].groupby("topic").agg({"pred": "mean", "efficacy":"max", "description":"max"}).sort_values("pred")
#%%

topic = 16

filtered_urls = df[df.topic == topic].domain.drop_duplicates()
fv = v.merge(filtered_urls, on="domain", how="inner")
fe = e.merge(fv.rename(columns={"ccid": "from_ccid"}), on="from_ccid", how="inner").merge(
    fv.rename(columns={"ccid": "to_ccid"}), on="to_ccid", how="inner")
G = nx.DiGraph()
G.add_nodes_from(fv.apply(lambda row: (row.ccid, dict(domain=row.domain)), axis=1))
G.add_edges_from(fe.apply(lambda row: (row.from_ccid, row.to_ccid), axis=1))
[(G.nodes()[k]["domain"],v) for k, v in sorted(nx.pagerank(G).items(), key=lambda x: -x[1])][:40]

#%%
# df["match"] = (df2.prob.gt(.5).astype("long") * 2 - 1).eq(df2.efficacy).astype("float")
df["match"] = df.pred ==
df[df.efficacy != 0].groupby("topic").agg({"match": "mean", "efficacy":"max", "description":"max"}).sort_values("match")

#%%