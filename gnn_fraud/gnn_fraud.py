#%%
import pandas as pd
from tqdm import trange
import numpy as np
df = pd.read_parquet("./mdt5/output_Top1kBM25_2019_mt5_2019_base-med_with_text")
topics = pd.read_csv("./data/topics.csv", sep="\t", index_col="topic")
v = pd.read_csv("./data/filtered_vertices.tsv", sep="\t", header=None, names="ccid rdomain nhosts domain".split())
e = pd.read_csv("./data/filtered_edges.tsv", sep="\t", header=None, names="from_ccid to_ccid".split())

from tldextract import extract

#%%

counts = {}
for i in trange(1,52):
    topic = i
    filtered_urls = df[df.topic == topic].url.apply(lambda x: extract(x).domain + '.' + extract(x).suffix).drop_duplicates()
    filtered_urls.name = "domain"
    fv = v.merge(filtered_urls, on="domain", how="inner")
    fe = e.merge(fv.rename(columns={"ccid":"from_ccid"}), on="from_ccid", how="inner").merge(fv.rename(columns={"ccid":"to_ccid"}), on="to_ccid", how="inner")
    counts[topics.loc[i].query] = fe.shape[0]
counts