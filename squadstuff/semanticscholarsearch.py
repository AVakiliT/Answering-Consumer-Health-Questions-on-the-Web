import time

import pandas as pd
import requests
from tqdm import tqdm
import json

# %%
topics = pd.read_csv("./data/topics_fixed.tsv.txt", sep="\t")
# %%
# all_papers = []
# for row in tqdm(topics.iterrows(), total=topics.shape[0]):
#     q = row[1]["query"]
#     t = row[1].topic
#     url = f"http://api.semanticscholar.org/graph/v1/paper/search?query={'+'.join(q.split())}&limit=100&fields=title,authors,abstract,year,citationCount,influentialCitationCount"
#     response = requests.get(url)
#     time.sleep(2)
#     all_papers.append((t, json.loads(response.content)))
#
# # %%
# datas = []
# for topic, topic_papers in tqdm(all_papers):
#     for data in topic_papers['data']:
#         datas.append((topic, *data.values()))
#
# df = pd.DataFrame(datas,
#                   columns=['topic', 'paperId', 'title', 'abstract', 'year', 'citationCount', 'influentialCitationCount',
#                            'authors'])

# df.to_parquet("data/semanticscholar.parquet")
df = pd.read_parquet("data/semanticscholar.parquet")
# %%
# stuff = []
# # def f(idx):
# for idx in tqdm(df.paperId):
#     url = f"https://api.semanticscholar.org/graph/v1/paper/{idx}?fields=fieldsOfStudy,tldr,embedding"
#     response = requests.get(url)
#     time.sleep(.1)
#     x = json.loads(response.content)
#     stuff.append({"paperId": idx, **x})
#%%
import collections

import numpy as np
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, \
    default_data_collator, AutoModelForSequenceClassification

squad_v2 = True
model_checkpoint = "./transformer_models/boolqstuff"
batch_size = 8

from datasets import load_dataset, load_metric, Dataset


#%%

max_length = 2000 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModelForSequenceClassification.from_pretrained("./transformer_models/boolqstuff")

df = df.merge(topics["topic description efficacy".split()], on="topic", how="inner")
df = df[~ df.abstract.isnull() & df.topic.eq(1)].reset_index(drop=True)
dataset = Dataset.from_pandas(df)

def tokenize_dataset(examples):
    tokenized_examples = tokenizer(
        examples["description"],
        examples["abstract"],
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
    )
    return tokenized_examples

tokenized_dataset = dataset.map(
    tokenize_dataset,
    batched=True,
    remove_columns=dataset.column_names

)

#%%
model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)

data_collator = default_data_collator
trainer = Trainer(
    model,
    args,
    eval_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

a = trainer.predict(tokenized_dataset)

#%%
from scipy.special import softmax
df["probs"] = [i for i in softmax(a.predictions, 1)]
df["prob_pos"] = df.probs.apply(lambda x: x[2])
df["prob_myb"] = df.probs.apply(lambda x: x[1])
df["prob_neg"] = df.probs.apply(lambda x: x[0])

#%%
df.groupby("topic").prob_pos.describe()