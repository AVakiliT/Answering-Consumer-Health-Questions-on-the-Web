import pandas as pd
import lightgbm as lgb

df = pd.read_parquet("./data/qrel_2021_1p_sentences_with_probs")
df["prob_pos"] = df.probs.apply(lambda x: x[2])
df["prob_neg"] = df.probs.apply(lambda x: x[0])

#%%
df["prob_pos_z"] = (df.prob_pos - df.prob_pos.mean())/df.prob_pos.std()
df["prob_neg_z"] = (df.prob_neg - df.prob_neg.mean())/df.prob_neg.std()
df["max_prob"] = df["prob_pos_z prob_neg_z".split()].abs().max(1)
temp = df.loc[df.groupby("topic docno".split()).max_prob.idxmax()]

#%%
X = df["prob_pos prob_neg host".split()]
X.host = X.host.astype("category")
y = df.score
group = df.topic
dataset = lgb.Dataset(X, label=y, weight=None, group=group, categorical_feature="auto")
param = {'num_leaves': 31, 'objective': 'binary'}
param['metric'] = ['auc', 'binary_logloss', 'map']
lgb.cv(param, dataset, 10, nfold=5)