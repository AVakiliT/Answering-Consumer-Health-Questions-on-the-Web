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
if __name__ == '__main__':

#%%

    df = pd.read_parquet("./mf/run.passage_6_3.boolq_logits.host_max.parquet")
    # df = df.merge(pd.read_csv("./data/topics.tsv", sep="\t")["topic description".split()], on="topic", how="inner")


    # topics = pd.read_csv("data/topics_fixed_extended.tsv.txt", sep="\t")
    # df = df.merge(topics["topic efficacy".split()], on="topic", how="inner")
    df = df[df.efficacy != 0]
    # user_ratings = topics[topics.efficacy != 0].apply(lambda x: pd.Series(
    #     [x.topic, "USER", float(x.efficacy == -1), float(x.efficacy == 1), x.efficacy],
    #     index=df.columns
    # ), axis=1)
    # df = pd.concat([df, user_ratings])


    df["pos"] = df.prob_pos.ge(.3).astype("float")
    df["neg"] = df.prob_neg.ge(.3).astype("float")
    # df["pred"] = df.apply(lambda x: np.array([x.prob_neg, x.prob_pos]).argmax(), axis=1)
    df["pred"] = df.apply(lambda x: np.array([x.prob_neg, x.prob_pos]).argmax(), axis=1)
    df.pred = df.pred * 2 -1
    df.pred = df.pred * df.apply(lambda x: np.array([x.prob_neg, x.prob_pos]).max(), axis=1)
    # df.loc[df.pred.lt(.3) & df.pred.gt(-0.3), "pred"] = 0
    df.pred = df.pred.astype("float32")

    # df.topic = df.topic.astype("category")
    # df["topic_id"] = df.topic.cat.codes
    df.host = df.host.astype("category")
    df["host_id"] = df.host.cat.codes




    # X_train = df[~(df.host.eq("USER")) & df.pred.ne(0)]["host_id topic".split()].to_numpy()
    # y_train = df[~(df.host.eq("USER")) & df.pred.ne(0)].pred.to_numpy()
    # # X_train = np.vstack([X_train, np.vstack([df[df.host.eq("USER") & df.topic.le(51) & df.pred.ne(0)]["host_id topic".split()].to_numpy()] * 10)])
    # # y_train = np.hstack([y_train, np.hstack([df[df.host.eq("USER") & df.topic.le(51) & df.pred.ne(0)].pred.to_numpy()] * 10)])
    # X_test = df[(df.host.eq("USER") & df.topic.ge(100))]["host_id topic".split()].to_numpy()
    # y_test = df[(df.host.eq("USER") & df.topic.ge(100))].pred.to_numpy()


    X = df[df.pred.ne(0)]["host_id topic".split()].to_numpy()
    y = df[df.pred.ne(0)].pred.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)
    #%%
    from torch import from_numpy
    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset
    from torch.utils.data import BatchSampler
    from torch.utils.data import RandomSampler

    def dataloader(*arrs, batch_size=32):
        dataset = TensorDataset(*arrs)
        bs = BatchSampler(RandomSampler(dataset),
                          batch_size=batch_size, drop_last=False)
        return DataLoader(dataset, batch_sampler=bs, num_workers=8)


    train = dataloader(from_numpy(X_train), from_numpy(y_train))
    test = dataloader(from_numpy(X_test), from_numpy(y_test))


    #%%
    def l2_regularize(array):
        return torch.sum(array ** 2.0)


    class MF(AbstractModel):
        def __init__(self, n_user, n_item, k=18, c_vector=1.0, batch_size=128):
            super().__init__()
            # These are simple hyperparameters
            self.k = k
            self.n_user = n_user
            self.n_item = n_item
            self.c_vector = c_vector
            self.batch_size = batch_size
            self.save_hyperparameters()

            # These are learned and fit by PyTorch
            self.user = nn.Embedding(n_user, k)
            self.item = nn.Embedding(n_item, k)

        def forward(self, inputs):
            # This is the most import function in this script
            # These are the user indices, and correspond to "u" variable
            user_id = inputs[:, 0]
            # Item indices, correspond to the "i" variable
            item_id = inputs[:, 1]
            # vector user = p_u
            vector_user = self.user(user_id)
            # equivalent:
            # self.user.weight[user_id, :]
            # vector item = q_i
            vector_item = self.item(item_id)
            # this is a dot product & a user-item interaction: p_u * q_i
            # shape vector_user is (batch_size, k)
            # vector_user * vector_item is shape (batch_size, k)
            # sum(vector_user * vector_item is shape, dim=1) (batch_size)
            ui_interaction = torch.sum(vector_user * vector_item, dim=1)
            return ui_interaction

        def loss(self, prediction, target):
            # MSE error between target = R_ui and prediction = p_u * q_i
            # target is (batchsize, 1)
            # target.squeeze (batchsize, )
            loss_mse = F.mse_loss(prediction, target.squeeze())
            return loss_mse, {"mse": loss_mse}

        def reg(self):
            # Compute L2 reularization over user (P) and item (Q) matrices
            reg_user = l2_regularize(self.user.weight) * self.c_vector
            reg_item = l2_regularize(self.item.weight) * self.c_vector
            # Add up the MSE loss + user & item regularization
            log = {"reg_user": reg_user, "reg_item": reg_item}
            total = reg_user + reg_item
            return total, log

    n_user = df.host.cat.categories.__len__()
    n_item = df.topic.max() + 1
    batch_size = 1024
    k = 32
    c_vector = 1e-5
    model = MF(n_user, n_item, k=k, c_vector=c_vector,
              batch_size=batch_size)

    trainer = pl.Trainer(max_epochs=100,
                         gpus=0,
                         log_every_n_steps=1,
                         progress_bar_refresh_rate=1,
                         )

    #%%
    trainer.fit(model, train, test)

    #%%
    results = trainer.test(model, test)

#%%
from sklearn.linear_model import LogisticRegression, Lasso
import lightgbm as lgbm
# df.pred = ((df.prob_pos - df.prob_pos.mean())/df.prob_pos.std()) * df.pos - (df.prob_neg-df.prob_neg.mean())/df.prob_neg.std() * df.neg

df = df[df.efficacy.ne(0)]

df.topic = df.topic.astype(int)
topics = df[df.efficacy.ne(0)].groupby("topic").topic.max().reset_index(drop=True)
df.topic = df.topic.astype("category")
df["topic_id"] = df.topic.cat.codes

m = coo_matrix((df.pred, (df.topic_id, df.host_id)), shape=(df.topic_id.max() + 1, df.host_id.max()+1))
m = np.array(m.todense())
y = df[df.efficacy.ne(0)].groupby("topic").efficacy.max().clip(lower=0).to_numpy()

# df.topic = df.topic.astype(int)

# train_index = topics.index[topics.astype(int).ge(1000) | topics.astype(int).le(51)]
# test_index = topics.index[topics.astype(int).ge(101) & topics.astype(int).le(150)]

train_index = topics.index[topics.astype(int).ge(1000) | topics.astype(int).ge(101)]
test_index = topics.index[topics.astype(int).ge(1) & topics.astype(int).le(51)]

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

y = df[df.efficacy.ne(0)].groupby("topic").efficacy.max().clip(lower=0).to_numpy()
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

np.mean(a)

#%%
clf = LogisticRegression()
clf.fit(m,y)
a = pd.DataFrame(sorted(list(zip((np.std(m, 0)*clf.coef_)[0].tolist(),df.host.cat.categories)), key=lambda x: abs(x[0]), reverse=True))
