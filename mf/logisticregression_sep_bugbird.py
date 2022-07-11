import lightgbm
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from mf.mf_modules import AbstractModel

#%%


#%%
from utils.util import url2host

dfo = pd.read_parquet(f"data/RunBM25.1k.passages_bigbird.top_mt5")
df3 = pd.read_parquet("./data/run.passages_bigbird.qapubmed_sep-logits")
df3 = df3.rename(columns={"sentence_scores": "sentence_score", "sentences": "sentences"})
#%%
# xxx = df3.groupby("topic docno".split()).apply(lambda x: x.loc[x.sentence_score.idxmax()]["prob_pos prob_may prob_neg sentence_score".split()])
xxx = df3.groupby("topic docno".split()).apply(lambda x: pd.Series({"prob_pos": (x.prob_pos * x.sentence_score).mean() - (x.prob_neg * x.sentence_score).mean()}))
xxx = xxx.reset_index()
df = dfo.merge(xxx, on='topic docno'.split(), how="left")
df = df.sort_values("topic score".split(), ascending=[True, False])
df.prob_pos = df.prob_pos.fillna(0)
# df.prob_neg = df.prob_neg.fillna(0)
# df.prob_may = df.prob_may.fillna(1)
# df.sentence_score = df.prob_may.fillna(0)
df = df[df.efficacy != 0]


#%%
# df["pred"] = df.apply(lambda x: np.array([x.prob_neg, x.prob_pos]).argmax(), axis=1)
# df["pred"] = df.apply(lambda x: x.prob_pos.gt(.33), axis=1)
df["pred"] = df.prob_pos
# df.pred = ((df.prob_pos* 2 -1))
# df.pred = df.pred * df.apply(lambda x: np.array([x.prob_neg, x.prob_pos]).max(), axis=1)
# df.loc[df.pred.lt(.3) & df.pred.gt(-0.3), "pred"] = 0
df.pred = df.pred.astype("float32")

# df.topic = df.topic.astype("category")
# df["topic_id"] = df.topic.cat.codes
df["host"] = df.url.apply(url2host)
df.host = df.host.astype("category")
df["host_id"] = df.host.cat.codes

#%%
from sklearn.linear_model import LogisticRegression, Lasso

df.topic = df.topic.astype(int)
topics = df.groupby("topic").topic.max().reset_index(drop=True)
df.topic = df.topic.astype("category")
df["topic_id"] = df.topic.cat.codes

m = coo_matrix((df.pred, (df.topic_id, df.host_id)), shape=(df.topic_id.max() + 1, df.host_id.max()+1))
m = np.array(m.todense())
y = df[df.efficacy.ne(0)].groupby("topic").efficacy.max().clip(lower=0).to_numpy()

# df.topic = df.topic.astype(int)

train_index = topics.index[topics.astype(int).ge(1000) | topics.astype(int).le(51)]
test_index = topics.index[topics.astype(int).ge(101) & topics.astype(int).le(150)]

# train_index = topics.index[topics.astype(int).ge(1000) | topics.astype(int).ge(101)]
# test_index = topics.index[topics.astype(int).ge(1) & topics.astype(int).le(51)]

X_train = m[train_index]
y_train = y[train_index]
X_test = m[test_index]
y_test = y[test_index]
# clf = LogisticRegression(penalty='l1', solver='liblinear').fit(X_train, y_train)
clf = Pipeline([
    # ('feature_selection', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear'))),
    # ('feature_selection', SelectFromModel(LogisticRegression(penalty='none'))),
    ('classification', LogisticRegression(penalty='l1', solver='liblinear'))
    # ('classification', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=200))
    # ('classification', LogisticRegression())
])
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
# pd.DataFrame(list(zip(topics[topics.ge(1) & topics.le(51)].to_list(), clf.predict(X_test).tolist()))).merge(topics)
#%%
from sklearn.linear_model import LogisticRegression, Lasso

y = df.groupby("topic").efficacy.max().clip(lower=0).to_numpy()
df.topic = df.topic.astype("category")
df["topic_id"] = df.topic.cat.codes
df = df[df.efficacy.ne(0)]
# df.pred = df.prob_pos * 2 - 1
m = coo_matrix((df.pred, (df.topic_id, df.host_id)), shape=(df.topic_id.max() + 1, df.host_id.max()+1))
m = np.array(m.todense())
kf = StratifiedKFold(n_splits=10, shuffle=False)

a = []
for train_index, test_index in kf.split(m,y):
    X_train = m[train_index]
    y_train = y[train_index]
    # y_test = df[df.efficacy.ne(0) & df.topic.ge(101)].groupby("topic").efficacy.max().clip(lower=0)
    X_test = m[test_index]
    y_test = y[test_index]
    # clf = LogisticRegression(penalty='l1', solver='liblinear').fit(X_train, y_train)
    clf = Pipeline([
        # ('feature_selection', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear'))),
        ('classification', LogisticRegression())
        # ('classification', LogisticRegression(penalty='l1', solver='liblinear')),
        # ('classification', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=200))
    ])
    clf.fit(X_train, y_train)
    a.append(clf.score(X_test, y_test))

print(np.mean(a))

#%%
# clf = LogisticRegression()
# clf.fit(m,y)
# a = pd.DataFrame(sorted(list(zip((np.std(m, 0)*clf.coef_)[0].tolist(),df.host.cat.categories)), key=lambda x: abs(x[0]), reverse=True))



