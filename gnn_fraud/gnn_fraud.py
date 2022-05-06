# %%
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import trange, tqdm
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer
from tldextract import extract
from boolq.bert_modules import BoolQBertModule
import networkx as nx

from gnn_fraud.fraud_utils import HMIDataset

df = pd.read_parquet("./mdt5/output_Top1kBM25_2019_mt5_2019_base-med_with_text")
topics = pd.read_csv("./data/topics.csv", sep="\t", index_col="topic")
v = pd.read_csv("./data/filtered_vertices.tsv", sep="\t", header=None, names="ccid rdomain nhosts domain".split())
e = pd.read_csv("./data/filtered_edges.tsv", sep="\t", header=None, names="from_ccid to_ccid".split())
df["domain"] = df.url.apply(lambda x: extract(x).domain + '.' + extract(x).suffix)
df = df.merge(topics["description efficacy".split()], on="topic", how="inner")
#%%

topic = 7

filtered_urls = df[df.topic == topic].domain.drop_duplicates()
fv = v.merge(filtered_urls, on="domain", how="inner")
fe = e.merge(fv.rename(columns={"ccid": "from_ccid"}), on="from_ccid", how="inner").merge(
    fv.rename(columns={"ccid": "to_ccid"}), on="to_ccid", how="inner")
G = nx.DiGraph()
G.add_nodes_from(fv.apply(lambda row: (row.ccid, dict(domain=row.domain)), axis=1))
G.add_edges_from(fe.apply(lambda row: (row.from_ccid, row.to_ccid), axis=1))
# %%

# counts = {}
# for topic in trange(1, 52):
#     filtered_urls = df[df.topic == topic].domain.drop_duplicates()
#     fv = v.merge(filtered_urls, on="domain", how="inner")
#     fe = e.merge(fv.rename(columns={"ccid": "from_ccid"}), on="from_ccid", how="inner").merge(
#         fv.rename(columns={"ccid": "to_ccid"}), on="to_ccid", how="inner")
#     counts[topics.loc[topic].description] = fe.shape[0]
# counts

# %%
NUM_CLASSES =  3
YES = "▁yes"
NO = "▁no"
IRRELEVANT = "▁irrelevant"
CHECKPOINT_PATH = f"checkpoints/boolq-simple/deberta-base-num_class=2-lr=1e-5-batch_size=16"
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=NUM_CLASSES).to(0)
lightning_module = BoolQBertModule.load_from_checkpoint(
    "./checkpoints/boolq-qrel/deberta-base-lr=1e-05-batch_size=16-aug/epoch=02-valid_F1=0.902-valid_Accuracy=0.902.ckpt",
    # "../checkpoints/boolq-simple/deberta-base-num_class=2-lr=1e-05-batch_size=16/epoch=04-valid_F1=0.818-valid_Accuracy=0.818.ckpt",
    tokenizer=tokenizer,
    model=model,
    save_only_last_epoch=True,
    num_classes=NUM_CLASSES,
    labels_text=[NO, IRRELEVANT, YES],
)


# %%
def prep_bert_sentence(q, p):
    return f"{q} [SEP] {p}"


df.apply(lambda x: f"{x.description} [SEP] {x.passage}", axis=1)

df["source_text"] = df.apply(lambda x: f"{x.description} [SEP] {x.passage}", axis=1)


# %%



dataset = HMIDataset(df, tokenizer=tokenizer)

# %%
data_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
)
#%%
probs = []
embeddings = []
for batch in tqdm(data_loader):
    with torch.no_grad():
        model.eval()
        a = model._modules['deberta'](input_ids=batch["input_ids"].to(0), attention_mask=batch["attention_mask"].to(0))
        a = model._modules['pooler'](a.last_hidden_state)
        logits = model._modules['classifier'](a)
        embeddings.append(a.cpu())
        probs.append(logits.cpu().softmax(-1))
embeddings = torch.cat(embeddings)
probs = torch.cat(probs)
torch.save(embeddings, "gnn_fraud/embeddings_2019.pt")
df["emb"] = pd.Series([x.numpy() for x in embeddings])
df["probs"] = pd.Series([x.numpy() for x in probs])
# with torch.no_grad():
#     a = model(input_ids=batch["input_ids"].to(0), attention_mask=batch["attention_mask"].to(0))
# a.logits.softmax(-1)[:,1]


# %%

# preds = []
# probs = []
# for batch in tqdm(data_loader):
#     with torch.no_grad():
#         model.eval()
#         a = model(input_ids=batch["input_ids"].to(0), attention_mask=batch["attention_mask"].to(0)).logits.softmax(-1)
#         probs.append(a.cpu())
#         preds.append(a.cpu().gt(.5).long())
# probs = torch.cat(probs)
# preds = torch.cat(preds)
# df["prob"] = pd.Series([x.item() for x in probs])
# df["pred"] = pd.Series([x.item() for x in preds])
df.to_parquet("./gnn_fraud/temp_df_aug")
df2 = pd.read_parquet("./gnn_fraud/old_temp_df")

#%%

# df["match"] = (df2.prob.gt(.5).astype("long") * 2 - 1).eq(df2.efficacy).astype("float")
df["match"] = df.probs.apply(lambda x: x.argmax()).sub(1).eq(df.efficacy)
df[df.efficacy != 0].groupby("topic").agg({"match": "mean", "efficacy":"max", "description":"max"}).sort_values("match")

#%%
df2[df2.efficacy != 0].groupby("topic").agg({"match": "mean", "efficacy":"max", "description":"max"}).sort_values("match")
