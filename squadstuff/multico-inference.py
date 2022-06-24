# dataset question context binary
from collections import Counter

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
from pathlib import Path
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from torch import nn
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, \
    default_data_collator, AutoModelForSequenceClassification, DataCollatorForTokenClassification, \
    AutoModelForTokenClassification, TrainerCallback, IntervalStrategy

# %%
from github.EncT5.enc_t5 import EncT5ForSequenceClassification, EncT5Tokenizer

window_size, step = 12, 6
# df = pd.read_parquet(f"/project/6004803/avakilit/Trec21_Data/data/RunBM25.1k.passages_{window_size}_{step}")

# df = pd.concat([
#     pd.read_parquet("/project/6004803/avakilit/Trec21_Data/data/Top1kBM25_2019"),
#     pd.read_parquet("/project/6004803/avakilit/Trec21_Data/data/Top1kBM25"),
# pd.read_parquet("/project/6004803/avakilit/Trec21_Data/Top1kRWBM25_32p")]).reset_index()

df = pd.read_parquet("data/RunBM25.1k.passages_12_6.top_mt5")
topics = pd.read_csv("data/topics_fixed_extended.tsv.txt", sep='\t')


# df2 = df["topic docno url text score".split()].merge(topics["topic description".split()], on="topic", how="inner")
dataset = Dataset.from_pandas(df)


# %%
max_length = 2048  # The maximum length of a feature (question and context)
doc_stride = 128  # The authorized overlap between two part of the context when splitting it is needed.

# model_checkpoint = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
# model_checkpoint = 'microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL'
# model_checkpoint = 'microsoft/deberta-base'
# model_checkpoint = 'l-yohai/bigbird-roberta-base-mnli'
model_checkpoint = 'google/bigbird-roberta-base'
model_name = model_checkpoint.split("/")[-1]
# model_checkpoint = f"checkpoints/{model_name}-mash-qa-tokenclassifier-binary-finetuned/best"
model_checkpoint = out_dir = f"checkpoints/{model_name}-mash-qa-tokenclassifier-binary-tokenchain-finetuned/best"
# model_checkpoint = 'distilbert-base-uncased'
# model_checkpoint = 'google/bigbird-pegasus-large-pubmed'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=2, ignore_mismatched_sizes=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})



# %%
def f(examples):
    # examples["question"] = [q.lstrip() for q in examples["question"]]
    # ss = tokenizer(examples['sentences'], add_epscial_toekns=False)
    # q = tokenizer(examples['question'])
    # sl = [len(s) for s in ss]
    tokenized_examples = tokenizer(
        examples["description"],
        examples["passage"],
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
    )
    return tokenized_examples


tokenized_datasets = dataset.map(f, batched=True,num_proc=2)
# %%
# disk_path = 'data/mahqa_classification_tokenized'
# Path(disk_path).mkdir(parents=True, exist_ok=True)
#
# tokenized_datasets.save_to_disk(disk_path)
# #%%
# tokenized_datasets = DatasetDict.load_from_disk(disk_path)
# %%
batch_size = 96

args = TrainingArguments(
    f"checkpoints/{model_name}-mash-qa-tokenclassifier-binary-finetuned",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
)



# %%



data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)





trainer: Trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# %%

#%%
x = tokenized_datasets.filter(lambda x: x['topic'] == 1, batched=True)
pred_x = trainer.predict(x)
pred = trainer.predict( tokenized_datasets)
torch.save(pred, 'data/tmp_pred_multico',pickle_protocol=4)
pred = torch.load('data/tmp_pred_multico',)