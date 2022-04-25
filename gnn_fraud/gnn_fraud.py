# %%
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import trange, tqdm
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer
from tldextract import extract
from boolq.bert_modules import BoolQBertModule

df = pd.read_parquet("./mdt5/output_Top1kBM25_2019_mt5_2019_base-med_with_text")
topics = pd.read_csv("./data/topics.csv", sep="\t", index_col="topic")
v = pd.read_csv("./data/filtered_vertices.tsv", sep="\t", header=None, names="ccid rdomain nhosts domain".split())
e = pd.read_csv("./data/filtered_edges.tsv", sep="\t", header=None, names="from_ccid to_ccid".split())
df["domain"] = df.url.apply(lambda x: extract(x).domain + '.' + extract(x).suffix)
df = df.merge(topics["description efficacy".split()], on="topic", how="inner")
#%%
import networkx as nx
topic = 1
filtered_urls = df[df.topic == topic].domain.drop_duplicates()
fv = v.merge(filtered_urls, on="domain", how="inner")
fe = e.merge(fv.rename(columns={"ccid": "from_ccid"}), on="from_ccid", how="inner").merge(
    fv.rename(columns={"ccid": "to_ccid"}), on="to_ccid", how="inner")
G = nx.DiGraph()
G.add_nodes_from(fv.apply(lambda row: (row.ccid, dict(domain=row.domain)), axis=1))
G.add_edges_from(fe.apply(lambda row: (row.from_ccid, row.to_ccid), axis=1))
[(G.nodes()[i]["domain"], v) for i, v in sorted(G.in_degree, key=lambda x: x[1], reverse=True)[:20]]
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
YES = "▁yes"
NO = "▁no"
IRRELEVANT = "▁irrelevant"
CHECKPOINT_PATH = f"checkpoints/boolq-simple/deberta-base-num_class=2-lr=1e-5-batch_size=16"
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=2).to(0)
lightning_module = BoolQBertModule.load_from_checkpoint(
    "./checkpoints/boolq-simple/deberta-base-num_class=2-lr=1e-05-batch_size=16/epoch=04-valid_F1=0.818-valid_Accuracy=0.818.ckpt",
    tokenizer=tokenizer,
    model=model,
    save_only_last_epoch=True,
    num_classes=2,
    labels_text=[NO, IRRELEVANT, YES],
)


# %%
def prep_bert_sentence(q, p):
    return f"{q} [SEP] {p}"


df.apply(lambda x: f"{x.description} [SEP] {x.passage}", axis=1)

df["source_text"] = df.apply(lambda x: f"{x.description} [SEP] {x.passage}", axis=1)


# %%
class HMIDataset(Dataset):
    """  PyTorch Dataset class  """

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: PreTrainedTokenizer,
            source_max_token_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len

    def __len__(self):
        """ returns length of data """
        return len(self.data)

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5/MT5 model"""

        data_row = self.data.iloc[index]
        source_text_encoding = self.tokenizer(
            data_row["source_text"],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return dict(
            input_ids=source_text_encoding["input_ids"].flatten(),
            attention_mask=source_text_encoding["attention_mask"].flatten(),
            efficacy=data_row.efficacy.flatten(),
            source=data_row.source_text,
            retrieval_score=data_row.score
        )


dataset = HMIDataset(df, tokenizer=tokenizer)

# %%
data_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
)
#%%

embeddings = []
for batch in tqdm(data_loader):
    with torch.no_grad():
        model.eval()
        a = model._modules['deberta'](input_ids=batch["input_ids"].to(0), attention_mask=batch["attention_mask"].to(0))
        a = model._modules['pooler'](a.last_hidden_state)
        embeddings.append(a.cpu())
embeddings = torch.cat(embeddings)
torch.save(embeddings, "gnn_fraud/embeddings_2019.pt")
# with torch.no_grad():
#     a = model(input_ids=batch["input_ids"].to(0), attention_mask=batch["attention_mask"].to(0))
# a.logits.softmax(-1)[:,1]

#%%

embeddings = []
for batch in tqdm(data_loader):
    with torch.no_grad():
        model.eval()
        a = model(input_ids=batch["input_ids"].to(0), attention_mask=batch["attention_mask"].to(0)).logits[:,1].gt()
        embeddings.append(a.cpu())
embeddings = torch.cat(embeddings)
df["emb"] = pd.Series([x for x in embeddings])
#%%

preds = []
for batch in tqdm(data_loader):
    with torch.no_grad():
        model.eval()
        a = model(input_ids=batch["input_ids"].to(0), attention_mask=batch["attention_mask"].to(0)).logits[:,1].gt(.5).long()
        preds.append(a.cpu())
preds = torch.cat(preds)