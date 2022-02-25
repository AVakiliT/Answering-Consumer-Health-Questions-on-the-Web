import pandas as pd
from tldextract import tldextract

urls = pd.read_csv("data/2019qrels.urls.csv", sep=" ")

urls["domain"] = urls.url.apply(lambda x: '.'.join(tldextract.extract(x)[-2:]))

qrels = pd.read_csv("./data/2019qrels.txt", sep=' ', header=None, index_col=False,
                    names=['topic_id', 'iteration', 'doc_id', 'relevance', 'stance', 'credibility'])

df = pd.merge(urls, qrels, left_on="id", right_on="doc_id", how="inner")[["domain", "credibility"]]

df = df[df.credibility >= 0]

df = df.groupby("domain").agg(count=("credibility","count"), credibility=("credibility", "mean"))


v = pd.read_csv("./data/filtered_vertices.tsv", sep="\t", header=None, names="ccid rdomain nhosts domain".split())

df2 = pd.merge(v, df, on="domain", how="inner")
