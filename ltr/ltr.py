import pandas as pd
import lightgbm as lgb
import numpy as np
qrels = pd.read_parquet("./data/qrel_2021_1p_sentences_with_probs")
top1krun = pd.read_parquet("./data/Top1kBM25_2021_1p_sentences_with_probs")


def f(df):
    df["prob_pos"] = df.probs.apply(lambda x: x[2])
    df["prob_neg"] = df.probs.apply(lambda x: x[0])
    df["prob_pos_z"] = (df.prob_pos - df.prob_pos.mean())/df.prob_pos.std()
    df["prob_neg_z"] = (df.prob_neg - df.prob_neg.mean())/df.prob_neg.std()
    df["max_prob"] = df["prob_pos_z prob_neg_z".split()].max(1)
    temp = df.loc[df.groupby("topic docno".split()).max_prob.idxmax()]
    return temp

qrel_top_passage = f(qrels)
top1k_top_passage = f(top1krun)
top1k_top_passage = top1k_top_passage.rename(columns={"score":"bm25"})

qrel_top_passage = qrel_top_passage.merge(pd.read_parquet('./data/qrels/2021_qrels_with_bm25.parquet')['topic docno bm25'.split()], on='topic docno'.split(), how='inner')
# original_qrels = pd.read_csv("data/qrels/2021_qrels.txt", names="topic iter docno usefulness stance credibility".split(), sep=" ")
# def unfixdocno(s):
#     return f"c4-{int(s[21:25]):04}-{int(s.split('.')[-1]):06}"
# original_qrels.docno = original_qrels.docno.apply(unfixdocno)
# qrel_top_passage = qrel_top_passage.merge(original_qrels['topic docno usefulness stance credibility'.split()], on='topic docno'.split(), how='inner')
qrel_top_passage["ranking"] = qrel_top_passage.groupby("topic").bm25.rank("dense", ascending=False)
#%%
qrel_top_passage = qrel_top_passage[qrel_top_passage.credibility >= 0]
X = qrel_top_passage["host".split()]
X.host = X.host.astype("category")
y = qrel_top_passage.credibility
y = y - y.min()
group = qrel_top_passage.topic
dataset = lgb.Dataset(X, label=y, weight=None, group=group, categorical_feature="auto")
# param = dict(objective='multiclass', num_class=3)
param = dict(objective='multiclass', num_class=3)
param['metric'] = ['multi_logloss']
# cv = lgb.cv(param, dataset, 10, nfold=5)


def construct_run(_lol):
    run = _lol["topic docno pred".split()]
    run = run.sort_values(by="topic pred".split(), ascending=[True, False])
    run.insert(2, "ranking", run.groupby("topic").pred.rank("dense", ascending=False).astype(int))
    run["tag"] = "ltr"
    run.insert(1, 'iter', 0)
    run.docno = run.docno.apply(lambda x: f"en.noclean.c4-train.{int(x.split('-')[1]):05}-of-07168.{int(x.split('-')[2])}")
    return run

from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=5)
runs = []
runs2 = []
stats = np.array([0.,0.])

for _x, _y in cv.split(X, y, group):
    train_data = lgb.Dataset(X.iloc[_x], label=y.iloc[_x], categorical_feature='auto', group=group.iloc[_x].to_frame(0).groupby(0)[0].count())
    eval_data = lgb.Dataset(X.iloc[_y], label=y.iloc[_y], categorical_feature='auto', group=group.iloc[_y].to_frame(0).groupby(0)[0].count())
    num_round = 10
    bst = lgb.train(param, train_data, num_round, valid_sets=[eval_data])
    # stats[0] += bst.best_score['valid_0']['ndcg@5']
    # stats[1] += bst.best_score['valid_0']['auc']

print((stats)/5)
#%%
    #make a run file from qrel validation set
    # ypred = bst.predict(X.iloc[_y])
    # lol = X.iloc[_y]
    # lol["pred"] = ypred
    # lol["efficacy"] = qrel_top_passage.iloc[_y].efficacy
    # lol["topic"] = qrel_top_passage.iloc[_y].topic
    # lol["docno"] = qrel_top_passage.iloc[_y].docno
    # lol["passage"] = qrel_top_passage.iloc[_y].passage
    # lol["score"] = qrel_top_passage.iloc[_y].score
    # lol = lol.sort_values(by="topic pred".split(), ascending=[True, False])
    # run = construct_run(lol)
    # runs.append(run)
    #
    # lol2 = top1k_top_passage[top1k_top_passage.topic.isin(qrel_top_passage.iloc[_y].topic.unique())]
    # X2 = lol2["prob_neg_z host".split()]
    # X2.host = X2.host.astype("category")
    # ypred2 = bst.predict(X2)
    # X2["pred"] = ypred2
    # X2["topic"] = lol2.topic
    # X2["docno"] = lol2.docno
    # X2["passage"] = lol2.passage
    # X2 = X2.sort_values(by="topic pred".split(), ascending=[True, False])
    # run2 = construct_run(X2)
    # runs2.append(run2)

#%%

runs = pd.concat(runs)
runs = runs.sort_values(by="topic pred".split(), ascending=[True, False])
runs.to_csv("./ltr/qrels_2021-kfold5-validation-run.txt", sep=" ", header=False, index=False)

runs2 = pd.concat(runs2)
runs2 = runs2.sort_values(by="topic pred".split(), ascending=[True, False])
runs2.to_csv("./ltr/Top1kBM25_2021-kfold5-validation-run.txt", sep=" ", header=False, index=False)

#%%
lgb.plot_tree(bst)
from matplotlib import pyplot as plt
plt.savefig("figures/figure.png", dpi=1200)
#%%
xxx = qrel_top_passage[qrel_top_passage.host.eq("www.webmd.com")]
xxx[xxx.efficacy.eq(1) & xxx.stance.eq(2)]
xxx[xxx.efficacy.eq(-1) & xxx.stance.eq(0)]