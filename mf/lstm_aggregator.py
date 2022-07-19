import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.sparse import coo_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from torch.nn import LSTM
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)
tqdm.pandas()
# %%


# %%
from utils.util import url2host

dfo = pd.read_parquet(f"data/RunBM25.1k.passages_bigbird.top_mt5")
dfo = dfo[dfo.efficacy != 0]
dfo["host"] = dfo.url.apply(url2host)

host_cat = dfo.host.astype("category")
host_mapping = {v: k for k, v in enumerate(host_cat.cat.categories)}
# df_topic_host["host_id"] = host_cat.cat.codes

df_topic_host = dfo.groupby("topic host".split()).progress_apply(lambda x: x.loc[x.score.idxmax()])
df_topic_host = df_topic_host.drop(columns=["host", "topic"])
df_topic_host = df_topic_host.sort_values("topic score".split(), ascending=[True, False])

df_sentence_logits = pd.read_parquet("./data/run.passages_bigbird.qapubmed_sep-logits2")
df_sentence_logits = df_sentence_logits.rename(columns={"sentence_scores": "sentence_score", "sentences": "sentence"})
df_sentence_logits = df_sentence_logits.merge(df_topic_host.reset_index()["topic host docno".split()], on="topic docno".split(), how="inner")
# df_sentence_logits = df_sentence_logits.drop(df_sentence_logits[df_sentence_logits.sentence_score.lt(0.80)].index)
MAX_LEN = 10
# %%
def func(x):
    sentences = x[x.index.isin(x.nlargest(MAX_LEN, "sentence_score").index)]
    sentences = sentences[sentences.sentence_score.gt(.85)]
    try:
        sentences = sentences[sentences.index.isin([sentences.sentence_score.idxmax()])]
    except:
        pass
    return pd.Series({
        "s": [np.array(
            sent.cls.tolist() + [sent.sentence_score, sent.prob_pos, sent.prob_may, sent.prob_neg]
        ) for sent in sentences.itertuples()]
    })
df_topic_host_features = df_sentence_logits.groupby("topic host docno".split()).progress_apply(func)
# df_sentence_logits = df_sentence_logits[df_sentence_logits.topic.le(105) & df_sentence_logits.topic.ge(100)]
# df_host_sentence_logits = df_sentence_logits.groupby("topic host docno".split()).progress_apply(
#     lambda x: x[x.index.isin(x.nlargest(MAX_LEN, "sentence_score").index)].drop(columns="topic host docno".split()).__len__())
#
# data = df_sentence_logits.groupby("topic host".split()).progress_apply(
#     lambda doc: pd.Series({
#         "s": [np.array(
#             sent.cls.tolist() + [sent.sentence_score, sent.prob_pos, sent.prob_may, sent.prob_neg]
#         ) for sent in doc.itertuples()]
#     })
# )
data = df_topic_host_features.join(df_topic_host["score efficacy".split()], how="inner")
data["host_id"] = data.index.map(lambda x: host_mapping[x[1]]).to_list()

data = data.reset_index().groupby("topic").progress_apply(lambda x: pd.Series({
    "s": x.s.to_list(),
    "mt5_score": x.score.to_list(),
    "host_id": x.host_id.to_list(),
    "labels": x.efficacy.max()
}))


def collate_fn(x):
    x = x[0]
    lol = [[j for j in i] for i in x["s"]]
    lol = [i + ((MAX_LEN - len(i)) * [np.zeros(772)]) for i in lol]
    lol = torch.tensor(np.array(lol), dtype=torch.float)
    return dict(
        hiddens=lol,
        mt5_scores=torch.tensor(x["mt5_score"], dtype=torch.float),
        host_ids=torch.tensor(x["host_id"], dtype=torch.long),
        labels=torch.tensor(x['labels'], dtype=torch.long)
    )

topics_2019 = list(range(1,51 + 1))
topics_2021 = list(range(101,150 + 1))
topics_r2 = list(range(1001,1090 + 1))
dl = DataLoader(Dataset.from_pandas(data[data.index.isin(topics_r2)]), batch_size=1, collate_fn=collate_fn)
dlt = DataLoader(Dataset.from_pandas(data[data.index.isin(topics_2021)]), batch_size=1, collate_fn=collate_fn)
#%%
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = LSTM(768, 100, bidirectional=True, batch_first=True)
        self.host_emb = nn.Embedding(num_embeddings=host_cat.cat.codes.max() + 1, embedding_dim=1)
        self.host = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1))
        self.sentence_lin = nn.Sequential(nn.Linear(200, 100), nn.ReLU(), nn.Linear(100, 1))
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, hiddens, mt5_scores, host_ids, labels):
        # host_score = self.host(self.host_emb(host_ids)).squeeze(-1)
        host_score = (self.host_emb(host_ids)).squeeze(-1)

        d2h = self.lstm(hiddens[:, :, :768])[0]
        d = self.sentence_lin(d2h).squeeze(-1)
        doc_score = d

        sentence_scores = hiddens[:, :, 768]
        sentence_pos = hiddens[:, :, 769]
        output = ((sentence_pos * sentence_scores).squeeze(-1).mean(-1) * host_score).mean()
        loss = self.loss(output, labels.float().clamp(min=0))
        return output, loss


model = Model().cuda()
optimizer = Adam(model.parameters(), lr=1e-2)

for i in range(10):
    losses = []
    accs = []
    pbar = tqdm(dl)
    for batch in pbar:
        optimizer.zero_grad()
        batchc = {k: v.cuda() for k, v in batch.items()}
        output, loss = model(**batchc)
        loss.backward()
        optimizer.step()
        loss_item = loss.detach().item()
        losses.append(loss_item)
        accs.append(output.detach().gt(0).eq(batch["labels"].gt(0)).item())
        pbar.set_description(f"[TRAI] loss: {np.mean(losses):.03} acc: {np.mean(accs):.03}")

    losses = []
    accs = []
    with torch.no_grad():
        pbar = tqdm(dlt)
        for batch in pbar:
            batchc = {k: v.cuda() for k, v in batch.items()}
            output, loss = model(**batchc)
            loss_item = loss.detach().item()
            losses.append(loss_item)
            accs.append(output.detach().gt(0).eq(batch["labels"].gt(0)).item())
            pbar.set_description(f"[TEST] loss: {np.mean(losses):.03} acc: {np.mean(accs):.03}")


