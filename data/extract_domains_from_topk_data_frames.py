# %%

# Take top k runs and get the urls
import sys

import pandas as pd
import tldextract

df = pd.concat(
    [pd.read_parquet(f"./data/Top1kBM25"),
     pd.read_parquet(f"./data/Top1kBM25_2019")])

domains = df.url.apply(lambda x: '.'.join(tldextract.extract(x)[-2:])).drop_duplicates()

r_domains = domains.apply(lambda x: '.'.join(x.split('.')[::-1]))

d = pd.concat({'domain': domains, 'rdomain': r_domains}, axis=1)
d.to_csv("./data/Top1kBM25_domains.tsv", sep="\t", index=False)

# %%
d = pd.read_csv("./data/Top1kBM25_domains.tsv", sep="\t")

v = pd.read_csv("./data/cc-main-2018-19-nov-dec-jan-domain-vertices.txt.gz", sep="\t",
                 names="id rdomain num_hosts".split())

e = pd.read_csv("./data/cc-main-2018-19-nov-dec-jan-domain-edges.txt.gz", sep="\t",
                 names="from_id to_id".split())

x = v.merge(d, on="rdomain", how="inner")

filtered_edges = e.merge(x, left_on="from_id", right_on="id")
filtered_edges = filtered_edges.merge(x, left_on="to_id", right_on="id")
filtered_edges[["from_id", "to_id"]].to_csv("./data/filtered_edges.tsv", sep="\t", index=False, header=False)
x.to_csv("./data/filtered_certices.tsv", sep="\t",  index=False, header=False)

