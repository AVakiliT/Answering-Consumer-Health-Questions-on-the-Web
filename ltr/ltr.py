import pandas as pd
import lightgbm as lgb

df = pd.read_parquet("./data/qrel_2021_1p_sentences_with_probs")
top1krun = pd.read_parquet("./data/Top1kBM25_2021_1p_sentences_with_probs")
def f(df):
    df["prob_pos"] = df.probs.apply(lambda x: x[2])
    df["prob_neg"] = df.probs.apply(lambda x: x[0])
    df["prob_pos_z"] = (df.prob_pos - df.prob_pos.mean())/df.prob_pos.std()
    df["prob_neg_z"] = (df.prob_neg - df.prob_neg.mean())/df.prob_neg.std()
    df["max_prob"] = df["prob_pos_z prob_neg_z".split()].abs().max(1)
    temp = df.loc[df.groupby("topic docno".split()).max_prob.idxmax()]
    return temp

temp = f(df)
temp2 = f(top1krun)

#%%
X = temp["prob_pos prob_neg host".split()]
X.host = X.host.astype("category")
y = temp.score
group = temp.topic
dataset = lgb.Dataset(X, label=y, weight=None, group=group, categorical_feature="auto")
param = {'num_leaves': 31, 'objective': 'binary'}
param['metric'] = ['auc', 'binary_logloss', 'map']
# cv = lgb.cv(param, dataset, 10, nfold=5)
top1krun = pd.read_parquet("./data/Top1kBM25_2021_1p_sentences_with_probs")

#%%
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
for _x, _y in cv.split(X, y, group):
    train_data = lgb.Dataset(X.iloc[_x], label=y.iloc[_x], categorical_feature=['host'], group=group.iloc[_x].to_frame(0).groupby(0)[0].count())
    eval_data = lgb.Dataset(X.iloc[_y], label=y.iloc[_y], categorical_feature=['host'], group=group.iloc[_y].to_frame(0).groupby(0)[0].count())
    param = {'num_leaves': 31}
    param['metric'] = 'auc map'.split()
    num_round = 10
    bst = lgb.train(param, train_data, num_round, valid_sets=[eval_data])


    #make a run file from qrel validation set
    ypred = bst.predict(X.iloc[_y])
    lol = X.iloc[_y]
    lol["pred"] = ypred
    lol["efficacy"] = temp.iloc[_y].efficacy
    lol["topic"] = temp.iloc[_y].topic
    lol["docno"] = temp.iloc[_y].docno
    lol["passage"] = temp.iloc[_y].passage
    run = construct_run(lol)
    runs.append(run)

    lol2 = temp2[temp2.topic.isin(temp.iloc[_y].topic.unique())]
    X2 = lol2["prob_pos prob_neg host".split()]
    X2.host = X2.host.astype("category")
    ypred2 = bst.predict(X2)
    X2["pred"] = ypred2
    X2["topic"] = lol2.topic
    X2["docno"] = lol2.docno
    run2 = construct_run(X2)
    runs2.append(run2)

#%%

runs = pd.concat(runs)
runs = runs.sort_values(by="topic pred".split(), ascending=[True, False])
runs.to_csv("./ltr/qrels_2021-kfold5-validation-run.txt", sep=" ", header=False, index=False)

runs2 = pd.concat(runs2)
runs2 = runs2.sort_values(by="topic pred".split(), ascending=[True, False])
runs2.to_csv("./ltr/Top1kBM25_2021-kfold5-validation-run.txt", sep=" ", header=False, index=False)