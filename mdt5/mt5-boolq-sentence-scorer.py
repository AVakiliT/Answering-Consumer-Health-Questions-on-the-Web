import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tldextract import extract
from boolq.bert_modules import BoolQBertModule
from gnn_fraud.fraud_utils import HMIDataset

print("Parsing args...", flush=True)
parser = argparse.ArgumentParser()
parser.add_argument("--topic_no", default=33, type=int)
parser.add_argument("--topic_file", default="/project/6004803/smucker/group-data/topics/misinfo-2021-topics.xml")
parser.add_argument("--model_type", default="base-med")
parser.add_argument("--bm25run",default="Top1kBM25_2019")
args = parser.parse_known_args()
topic_no = args[0].topic_no
df = pd.read_parquet(f"./data/{args[0].bm25run}_1p_sentences/")
df = df.rename(columns={"bm25":"score"})
df["domain"] = df.url.apply(lambda x: extract(x).domain + '.' + extract(x).suffix)
df["host"] = df.url.apply(lambda x: extract(x).subdomain + '.' + extract(x).domain + '.' + extract(x).suffix)
topics = pd.read_csv("./data/topics.csv", sep="\t", index_col="topic")
df = df.merge(topics["description efficacy".split()], on="topic", how="inner")
df["source_text"] = df.apply(lambda x: f"{x.description} [SEP] {x.passage}", axis=1)
#%%
df = df[df.topic == topic_no]

#%%
NUM_CLASSES =  3
YES = "▁yes"
NO = "▁no"
IRRELEVANT = "▁irrelevant"
# CHECKPOINT_PATH = f"checkpoints/boolq-simple/deberta-base-num_class=2-lr=1e-5-batch_size=16"
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=NUM_CLASSES).to(0)
lightning_module = BoolQBertModule.load_from_checkpoint(
    "./checkpoints/boolq-qrel/deberta-base-lr=1e-05-batch_size=16-aug/epoch=01-valid_F1=0.901-valid_Accuracy=0.901.ckpt",
    # "../checkpoints/boolq-simple/deberta-base-num_class=2-lr=1e-05-batch_size=16/epoch=04-valid_F1=0.818-valid_Accuracy=0.818.ckpt",
    tokenizer=tokenizer,
    model=model,
    save_only_last_epoch=True,
    num_classes=NUM_CLASSES,
    labels_text=[NO, IRRELEVANT, YES],
)

#%%
dataset = HMIDataset(df, tokenizer=tokenizer)

# %%
data_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
)

#%%
probs = []
# embeddings = []
for batch in tqdm(data_loader):
    with torch.no_grad():
        model.eval()
        a = model._modules['deberta'](input_ids=batch["input_ids"].to(0), attention_mask=batch["attention_mask"].to(0))
        a = model._modules['pooler'](a.last_hidden_state)
        logits = model._modules['classifier'](a)
        # embeddings.append(a.cpu())
        probs.append(logits.cpu().softmax(-1))
# embeddings = torch.cat(embeddings)
probs = torch.cat(probs)
# torch.save(embeddings, "gnn_fraud/embeddings_2019.pt")
df["probs"] = pd.Series([x.numpy() for x in probs])
out_path = f"./data/{args[0].bm25run}_1p_sentences_with_probs/topic-{topic_no}.snappy.parquet"
Path(out_path).parent.mkdir(exist_ok=True)
df.to_parquet(out_path)