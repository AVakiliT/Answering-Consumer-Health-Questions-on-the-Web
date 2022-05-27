import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression

qrels = pd.read_parquet("./data/qrel_2021_1p_sentences_with_probs")
top1krun = pd.read_parquet("./data/Top1kBM25_2021_1p_sentences_with_probs")
# top1krun_2019 = pd.read_parquet("./data/Top1kBM25_2019_1p_sentences_with_probs")


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
qrel_top_passage.host = qrel_top_passage.host.astype('category')
top1k_top_passage.host = top1k_top_passage.host.astype('category')

qrel_top_passage = qrel_top_passage.merge(pd.read_parquet('./data/qrels/2021_qrels_with_bm25.parquet')['topic docno bm25'.split()], on='topic docno'.split(), how='inner')
# original_qrels = pd.read_csv("data/qrels/2021_qrels.txt", names="topic iter docno usefulness stance credibility".split(), sep=" ")
# def unfixdocno(s):
#     return f"c4-{int(s[21:25]):04}-{int(s.split('.')[-1]):06}"
# original_qrels.docno = original_qrels.docno.apply(unfixdocno)
# qrel_top_passage = qrel_top_passage.merge(original_qrels['topic docno usefulness stance credibility'.split()], on='topic docno'.split(), how='inner')
qrel_top_passage["ranking"] = qrel_top_passage.groupby("topic").bm25.rank("dense", ascending=False)
#%%
qrel_mt5 = pd.read_parquet("./data/qrels.2021.passages_6_3.top_passage_mt5.parquet")
qrel_top_passage = qrel_top_passage.merge(qrel_mt5["topic docno mt5".split()], on="topic docno".split(), how="inner")

# top1krun_mt5 = pd.read_parquet("../data/")

#%% PAGERANK
pagerank_df = pd.read_csv(r"C:\Users\Amir\Downloads\cc-main-2018-19-nov-dec-jan-domain-ranks.txt.gz", sep="\t")
pagerank_df = pagerank_df.rename(columns={s : s.replace("#","") for s in pagerank_df.columns})
pagerank_df["domain"] = pagerank_df["host_rev"].apply(lambda x: '.'.join(x.split('.')[::-1]))
qrel_top_passage = qrel_top_passage.merge(pagerank_df["domain pr_val".split()], on="domain", how="left")
top1k_top_passage = top1k_top_passage.merge(pagerank_df["domain pr_val".split()], on="domain", how="left")
#%%
# def get_vs(row):
#     vp = np.zeros(len(cats))
#     vn = np.zeros(len(cats))
#     vp[cats.get_loc(row.host)] = max(vp[cats.get_loc(row.host)], row.stance.eq())
#     vn[cats.get_loc(row.host)] = max(vp[cats.get_loc(row.host)], row.prob_neg_z)
#     return np.concatenate([vp, vn])
#
# cats = qrel_top_passage.host.cat.categories
# qrel_top_passage["v"] = qrel_top_passage.apply(get_vs, axis=1)
#%%
# def calc_gain(x):
#     return ((x.efficacy.eq(-1) & x.stance.eq(0) & x.usefulness.gt(0)).sum() / ((x.efficacy.eq(-1) & x.usefulness.gt(0)).sum()+.01), x.efficacy.count())
#
# def calc_cumulative_gain(x):
#     x.

#%%
y = qrel_top_passage.score
y = y - y.min()
# criteria = [y.between(-3, -1), y.between(0, 4), y.between(5, 12)]
# values = [0, 1, 2]
# y = pd.Series(np.select(criteria, values, 0))
group = qrel_top_passage.topic
param = dict(
    objective='regression',
    metric='auc ndcg'.split(),
    # label_gain=np.array([-4, -2,    -1,    0,    1,    2,    4,    8,   16,   32,   64,  128, 256,  512, 1024, 2048])
             )
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
stats = [[],[]]

for index_train, index_test in cv.split(qrel_top_passage, y, group):
    X = qrel_top_passage["prob_pos prob_neg pr_val efficacy".split()]

    # v_in = np.vstack(qrel_top_passage.iloc[index_train].groupby("topic").v.sum().to_list())
    # v_y = qrel_top_passage.iloc[index_train].groupby('topic').efficacy.max().to_numpy()
    # clf = LogisticRegression(random_state=0).fit(v_in, v_y)
    # v_in_t = np.vstack(qrel_top_passage.iloc[index_test].groupby("topic").v.sum().to_list())
    # v_y_t = qrel_top_passage.iloc[index_test].groupby('topic').efficacy.max().to_numpy()
    # # stats[2] += clf.score(v_in_t, v_y_t)
    # print("cldscore", clf.score(v_in_t, v_y_t))
    train_data = lgb.Dataset(X.iloc[index_train], label=y.iloc[index_train], categorical_feature='auto', group=group.iloc[index_train].to_frame(0).groupby(0)[0].count())
    eval_data = lgb.Dataset(X.iloc[index_test], label=y.iloc[index_test], categorical_feature='auto', group=group.iloc[index_test].to_frame(0).groupby(0)[0].count())
    num_round = 10
    bst = lgb.train(param, train_data, num_round, valid_sets=[eval_data])
    stats[0] += [bst.best_score['valid_0']['ndcg@5']]
    stats[1] += [bst.best_score['valid_0']['auc']]
    print(qrel_top_passage.iloc[index_train].groupby("topic").efficacy.max().sum())
    print(np.mean(stats[0]), np.mean(stats[1]))

    # make a run file from qrel validation set
    ypred = bst.predict(X.iloc[index_test])
    lol = X.iloc[index_test]
    lol["pred"] = ypred
    lol["efficacy"] = qrel_top_passage.iloc[index_test].efficacy
    lol["topic"] = qrel_top_passage.iloc[index_test].topic
    lol["docno"] = qrel_top_passage.iloc[index_test].docno
    lol["passage"] = qrel_top_passage.iloc[index_test].passage
    lol["score"] = qrel_top_passage.iloc[index_test].score
    lol = lol.sort_values(by="topic pred".split(), ascending=[True, False])
    run = construct_run(lol)
    runs.append(run)

    lol2 = top1k_top_passage[top1k_top_passage.topic.isin(qrel_top_passage.iloc[index_test].topic.unique())]
    X2 = lol2["prob_pos prob_neg pr_val efficacy".split()]
    ypred2 = bst.predict(X2)
    X2["pred"] = ypred2
    X2["topic"] = lol2.topic
    X2["docno"] = lol2.docno
    X2["passage"] = lol2.passage
    X2 = X2.sort_values(by="topic pred".split(), ascending=[True, False])
    run2 = construct_run(X2)
    runs2.append(run2)



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