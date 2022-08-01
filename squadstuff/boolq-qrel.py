from collections import Counter

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
import torch
#%%
from squadstuff.boolq_utils import MyTrainer, compute_metrics, EvaluationCallback
from utils.util import topics_2019, topics_2021

df1 = pd.read_parquet("./qreldataset/2019_mt5_dataset.parquet").reset_index(drop=True)
df1 = df1.rename(columns={"stance": "credibility", "credibility": "stance"})
df1 = df1[df1.stance.ge(0)]

def gen_labels(row):
    if row.stance == 2 or row.stance == 0:
        return 1
    if row.stance == 3:
        return 2
    if row.stance == 1:
        return 0
    return float('nan')
df1["labels"] = df1.apply(gen_labels, axis=1)


df2 = pd.read_parquet("qreldataset/Qrels.2021.passages_6_3.top_mt5")

def gen_labels(row):
    if row.supportiveness == 2:
        return 2
    if row.supportiveness == 0:
        return 0
    if row.supportiveness == 1:
        return 1
    return float('nan')
df2 = df2[df2.supportiveness.ge(0)]
df2["labels"] = df2.apply(gen_labels, axis=1)

df2 = df2.rename(columns=dict(supportiveness="stance"))
df2 = df2.merge(pd.read_csv('./data/topics_fixed_extended.tsv.txt', sep='\t')["topic description".split()], on="topic", how="inner")
df1 = df1.rename(columns=dict(text="passage"))

df = pd.concat([df1, df2]).reset_index(drop=True)

train_idx = df[df.topic.isin(topics_2019)].index
test_idx = df[df.topic.isin(topics_2021)].index

# gss = GroupShuffleSplit(n_splits=1, train_size=.5, random_state=42)
# train_idx, test_idx = gss.split(df, groups=df.topic).__next__()
# train_idx = df.iloc[train_idx].index
# test_idx = df.iloc[test_idx].index
# %%
MAX_LENGTH = 512  # The maximum length of a feature (question and context)
# MODEL_START_POINT = 'blizrys/biobert-v1.1-finetuned-pubmedqa'
# MODEL_START_POINT = 'facebook/muppet-roberta-base'
# MODEL_START_POINT = 'blizrys/biobert-v1.1-finetuned-pubmedqa'
model_name = "biobert"
MODEL_START_POINT = f"checkpoints/pubmed_qa-biobert-v1.1-finetuned-pubmedqa/pubmedqa-phase-iii/best"
tokenizer = AutoTokenizer.from_pretrained(MODEL_START_POINT)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_START_POINT, num_labels=3)


x = tokenizer(
    df.description.to_list(),
    df.passage.to_list(),
    truncation="only_second",
    max_length=MAX_LENGTH,
)

d = pd.DataFrame.from_dict({i: x[i] for i in x.keys() if i != 'offset_mapping'})
d.index = df.index
d["labels"] = df["labels"].to_list()
d["topic"] = df["topic"].to_list()

tokenized_datasets = DatasetDict({
    "train": Dataset.from_pandas(d.loc[train_idx]),
    "test": Dataset.from_pandas(d.loc[test_idx])
})

class_weights = df.labels.value_counts().sort_index()
class_weights = (1 - np.array(list(class_weights.values)) / sum(class_weights.values)).astype('float')

# tokenized_datasets = tokenized_datasets.train_test_split(0.5, seed=42)
output_dir = 'checkpoints/boolq-qrel2'
batch_size = 32
params=dict(
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
)
args = TrainingArguments(
    output_dir,
    **params,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
)



#%%
trainer: Trainer = MyTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    class_weights=class_weights
)

trainer.add_callback(EvaluationCallback(output_dir=output_dir))

#%%
trainer.train()
trainer.evaluate()

#%%
# # trainer: Trainer = Trainer(
# #     AutoModelForSequenceClassification.from_pretrained(MODEL_START_POINT, num_labels=3),
# #     TrainingArguments(output_dir=output_dir,
# #                       save_strategy="no", evaluation_strategy="no",
# #                       **params,
# #                       ),
# #     train_dataset=concatenate_datasets([tokenized_datasets["train"],tokenized_datasets["test"]]),
# #     data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
# #     tokenizer=tokenizer
# # )
# # #%%
# # trainer.train()
# # trainer.save_model(Path(output_dir)/"best")