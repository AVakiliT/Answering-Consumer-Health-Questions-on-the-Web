import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold

from utils.util import url2host, url2domain

qrels = pd.read_parquet("./data/qrel_2021_1p_sentences_with_probs")
df = qrels[qrels.stance.ge(0)].groupby("topic host".split()).apply(
    lambda x: pd.Series([x.stance.mean() - 1], index=["stance"])).reset_index()
l = [i for i, x in qrels[qrels.stance.ge(0)].host.value_counts().iteritems() if x > 20]
df = df[df.host.isin(l)]
df.host = df.host.astype("category")
df["host_id"] = df.host.cat.codes
df.topic = df.topic.astype("category")
df["topic_id"] = df.topic.cat.codes
df = df.sort_values("topic_id host_id".split(), ascending=True)
m = coo_matrix((df.stance.astype("float"), (df.topic_id, df.host_id)),
               shape=(df.topic_id.max() + 1, df.host_id.max() + 1))
m = np.array(m.todense())

# %%
X = m
y = qrels.groupby("topic").efficacy.max().clip(lower=0).to_numpy()
clf = LogisticRegressionCV(cv=7, scoring='accuracy')
clf.fit(X, y)
clf.scores_[1].mean(axis=0)

# %%
a = []
kf = StratifiedKFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = LogisticRegression(penalty="none")
    clf = clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    a.append(clf.score(X_test, y_test))

np.mean(a)

#%%
h = (qrels[((qrels.stance.eq(2) & qrels.efficacy.eq(1)) | (qrels.stance.eq(0) & qrels.efficacy.eq(-1)))].host.value_counts() - \
qrels[((qrels.stance.eq(2) & qrels.efficacy.eq(-1)) | (qrels.stance.eq(0) & qrels.efficacy.eq(1)))].host.value_counts())\
    .sort_values(ascending=False).iloc[:100].index

#%%
df = pd.concat([pd.read_parquet("mdt5/output_Top1kBM25_2019_mt5_2019_base-med_with_text"),
                pd.read_parquet("mdt5/output_Top1kBM25_2021_mt5_2021_base-med_with_text")])
df = df.rename(columns={"docid": "docno"})
df = df.merge(pd.read_csv("./data/topics.tsv", sep="\t")["topic description efficacy".split()], on="topic", how="inner")
out_df = pd.read_parquet(["./mf/2019_passage_6_3.boolq_logits.parquet", "./mf/2021_passage_6_3.boolq_logits.parquet"])
df = df.merge(out_df, on="topic docno".split(), how="inner")
df["prob_pos"] = df.logits.apply(lambda x: torch.tensor(x).softmax(-1)[2].item())
df["prob_neg"] = df.logits.apply(lambda x: torch.tensor(x).softmax(-1)[0].item())

df["host"] = df.url.apply(url2host)
df["domain"] = df.url.apply(url2domain)
#%%
def g(d):
    a = d.sort_values("prob_neg", ascending=True)
    a["r"] = range(1, a.shape[0] + 1)
    a.r = 1 / a.r
    p = (a.host.isin(h) * a.r).sum()

    a = d.sort_values("prob_neg", ascending=False)
    a["r"] = range(1, a.shape[0] + 1)
    a.r = 1 / a.r
    n = (a.host.isin(h) * a.r).sum()

    m = (d.efficacy.max() >= 1) == ((p-n) >= 0)
    return p-n, m

#%%
df[df.topic.le(51) & df.efficacy.ne(0)].groupby("topic").apply(g)