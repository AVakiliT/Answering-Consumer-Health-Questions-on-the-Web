from collections import Counter

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

import json

from pathlib import Path

import pandas as pd

import torch

from datasets import Dataset, DatasetDict, concatenate_datasets

from torch import nn

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, \
 \
    default_data_collator, AutoModelForSequenceClassification, DataCollatorForTokenClassification, \
    AutoModelForTokenClassification, TrainerCallback, IntervalStrategy

# %%
# df = pd.concat([
#     pd.read_parquet("/project/6004803/avakilit/Trec21_Data/data/Top1kBM25_2019").reset_index(),
#     pd.read_parquet("/project/6004803/avakilit/Trec21_Data/data/Top1kBM25").reset_index(),
#     pd.read_parquet("/project/6004803/avakilit/Trec21_Data/Top1kRWBM25_32p").reset_index()
# ])
# df = pd.read_parquet("data/Top1kBM25.snappy.parquet")
# topics = pd.read_csv('./data/topics_fixed_extended.tsv.txt', sep='\t')
# df = df.merge(topics['topic description'.split()], on='topic', how='inner')
# df = df.sort_values("topic score".split(), ascending=[True, False])
# df.to_parquet("data/Top1kBM25.snappy.parquet")
# %%

max_length = 4096  # The maximum length of a feature (question and context)
doc_stride = 0  # The authorized overlap between two part of the context when splitting it is needed.

# model_checkpoint = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
# model_checkpoint = 'microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL'
# model_checkpoint = 'microsoft/deberta-base'
# model_checkpoint = 'l-yohai/bigbird-roberta-base-mnli'
model_checkpoint = 'google/bigbird-roberta-base'
model_name = model_checkpoint.split("/")[-1]
# model_checkpoint = f"checkpoints/{model_name}-mash-qa-tokenclassifier-binary-finetuned/best"
out_dir = f"checkpoints/{model_name}-mash-qa-tokenclassifier-binary-tokenchain-finetuned"
# model_checkpoint = 'distilbert-base-uncased'
# model_checkpoint = 'google/bigbird-pegasus-large-pubmed'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(out_dir + '/best', num_labels=2, ignore_mismatched_sizes=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# %%

# def f(examples):
#     # examples["question"] = [q.lstrip() for q in examples["question"]]
#     # ss = tokenizer(examples['sentences'], add_epscial_toekns=False)
#     # q = tokenizer(examples['question'])
#     # sl = [len(s) for s in ss]
#     tokenized_examples = tokenizer.batch_encode(
#         examples["description"],
#         examples["text"],
#         truncation="only_second",
#         max_length=max_length,
#         padding="max_length",
#         stride=doc_stride,
#         return_overflowing_tokens=True,
#         return_offsets_mapping=True,
#     )
#     return tokenized_examples


# dataset = Dataset.from_pandas(df['description text'.split()])
# df = df.sort_values("topic score".split(), ascending=[True, False])
k = 10000
xs = []
from tqdm import trange
for i in trange(0, df.shape[0], k):
    x = tokenizer(
        df.description[i:i + k].to_list(), df.text[i:i + k].to_list(),
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    xx = pd.DataFrame.from_dict({i: x[i] for i in x.keys()})
    xx["topic"] = xx.overflow_to_sample_mapping.apply(lambda x: df.topic.iloc[i + x])
    xx["docno"] = xx.overflow_to_sample_mapping.apply(lambda x: df.docno.iloc[i + x])
    xx['overflow_to_sample_mapping'] = xx['overflow_to_sample_mapping'] + i
    xs.append(xx)

xs = pd.concat(xs)
xs[['input_ids', 'attention_mask',
       'overflow_to_sample_mapping', 'topic', 'docno']].to_parquet('data/Top1kBM25_plus_description.tokenized.bigbird.4096.parquet')
# import spacy
# nlp = spacy.blank('en')
# nlp.add_pipe("sentencizer")
# docs = nlp.pipe(df.text.to_list(), n_process=28)
#
# temp = [' [SEP] '.join([sent.sent.text.strip() for sent in doc.sents]) for doc in tqdm(docs, total=df.shape[0])]
# def sentencize(text):
#
#     sentences =
#     return ' [SEP] '.join(sentences)

# x = tokenizer(
#     df.description.to_list(), df.text.to_list(),
#     truncation="only_second",
#     max_length=max_length,
#     # padding="max_length",
#     stride=doc_stride,
#     return_overflowing_tokens=True,
#     return_offsets_mapping=True,
# )
# x = pd.DataFrame.from_dict({i: x[i] for i in x.keys() if i != 'offset_mapping'})
# x.offset_mapping = x.offset_mapping.apply(lambda y: list(map(list,y)))
# x = pd.concat([pd.DataFrame.from_dict({i : x[i] for i in x.keys()}) for x in xs])
x.to_parquet('data/Top1kBM25_plus_description.tokenized.bigbird.4096.parquet')
# %%
x = pd.read_parquet('data/Top1kBM25_plus_description.tokenized.bigbird.4096.parquet')
tokenized_datasets = Dataset.from_pandas(x)
# tokenized_dataset = concatenate_datasets([Dataset.from_dict(x) for x in tqdm(xs)])
# %%
batch_size = 32

args = TrainingArguments(
    out_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer: Trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

pred_x = trainer.predict(tokenized_datasets)
torch.save(pred_x, 'tmp_bigbird', pickle_protocol=4)
# %%
import numpy as np
from tqdm import tqdm, trange
import re
from collections import defaultdict

passages = defaultdict(list)
for pred, example in tqdm(zip(pred_x.predictions, tokenized_datasets), total=len(pred_x.predictions)):
    passage = tokenizer.decode(np.array(example['input_ids'])[pred.argmax(-1) == 1])
    passages[example['overflow_to_sample_mapping']].append(passage)

torch.save(passages, 'tmp_bigbird2')