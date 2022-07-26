from collections import Counter
from multiprocess.managers import BaseManager, DictProxy

import spacy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

import json

from pathlib import Path
from multiprocess import Pool
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from datasets import Dataset, DatasetDict, concatenate_datasets

from torch import nn

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, \
 \
    default_data_collator, AutoModelForSequenceClassification, DataCollatorForTokenClassification, \
    AutoModelForTokenClassification, TrainerCallback, IntervalStrategy

# %%
# temp = pd.read_parquet("/project/6004803/avakilit/Trec21_Data/data/Top1kBM25_2019").reset_index()
# temp2 = pd.read_parquet("/project/6004803/avakilit/Trec21_Data/data/Top1kBM25").reset_index()
# temp3 = pd.read_parquet("/project/6004803/avakilit/Trec21_Data/data/Top1kBM25_2022_32p").reset_index()
# temp4 = pd.read_parquet("/project/6004803/avakilit/Trec21_Data/Top1kRWBM25_32p").reset_index()
# df = pd.concat([
# temp,temp2,temp3, temp4
# ])

# topics = pd.read_csv('./data/topics_fixed_extended.tsv.txt', sep='\t')
# df2 = df.merge(topics['topic description'.split()], on='topic', how='inner')
# df2 = df2.sort_values("topic score".split(), ascending=[True, False])
# df2 .to_parquet("data/Top1kBM25.snappy.parquet")
df = pd.read_parquet("data/Top1kBM25.snappy.parquet")
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
if True:
    def parallelize_dataframe(df, func, n_cores=25):
        df_split = np.array_split(df, n_cores * 8)
        pool = Pool(n_cores)
        df = pd.concat(pool.imap(func, tqdm(df_split)))
        pool.close()
        pool.join()
        return df

    nlp = spacy.blank('en')

    nlp.add_pipe("sentencizer")
    def ff(_df):
        return _df.apply(lambda x: ' [SEP] '.join([sent.sent.text.strip() for sent in nlp(x).sents]))
    xx = parallelize_dataframe(df.text, ff)
    df.text = xx
    # yy = df.apply(lambda row: f"[CLS] {row.description} [SEP] {row.text} [SEP]", axis=1)

    k = 10000
    xs = []
    from tqdm import trange
    for i in trange(0, df.shape[0], k):
        x = tokenizer(
            df.description[i:i + k].to_list(),
            df.text[i:i + k].to_list(),
            max_length=max_length,
            truncation="only_second",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            # add_special_tokens=False
        )
        xx = pd.DataFrame.from_dict({i: x[i] for i in x.keys()})
        xx["topic"] = xx.overflow_to_sample_mapping.apply(lambda x: df.topic.iloc[i + x])
        xx["docno"] = xx.overflow_to_sample_mapping.apply(lambda x: df.docno.iloc[i + x])
        xx['overflow_to_sample_mapping'] = xx['overflow_to_sample_mapping'] + i
        xs.append(xx)

    x = pd.concat(xs)
    x[['input_ids', 'attention_mask',
           'overflow_to_sample_mapping', 'topic', 'docno']].to_parquet('data/Top1kBM25_plus_description.sep_tokenized.bigbird.4096.parquet')
    # # tokenizer.decode(x.input_ids.iloc[123])
    print("reading tokenized data...")
x = pd.read_parquet('data/Top1kBM25_plus_description.sep_tokenized.bigbird.4096.parquet')

# %%
# x = pd.read_parquet('data/Top1kBM25_plus_description.tokenized.bigbird.4096.parquet')
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
torch.save(pred_x.predictions, 'tmp_bigbird', pickle_protocol=4)
pred_x = torch.load('tmp_bigbird')
# %%

import numpy as np
from tqdm import tqdm, trange
import re
from collections import defaultdict

# pool = Pool(processes=10)


from scipy.special import softmax

stuff = []
for pred, example in tqdm(zip(pred_x, x['topic input_ids docno'.split()].itertuples()), total=pred_x.shape[0]):
    token_idx = [False] * 4096
    sentence_idx = [False] * 4096
    current = False
    s = softmax(pred, -1)[:, -1]
    l = np.array(list(example.input_ids) + ([0] * (4096 - len(example.input_ids))))
    for i, (token, token_prediction) in enumerate(zip(l, pred.argmax(-1))):
        if token == 0:
            current = False
        elif token == 66 and token_prediction == 1:
            current = True
            if i != example.input_ids.shape[0] - 1:
                sentence_idx[i] = True
        elif token == 66 and token_prediction == 0:
            current = False
        elif token != 66:
            pass
        token_idx[i] = current
    passage = tokenizer.decode(l[token_idx])
    passage_sentence_scores = s[sentence_idx]
    sentences = passage.split('[SEP]')[1:]
    len_sentences = [i > 15 for i in map(len,sentences)]
    passage_sentence_scores = [s for s, i in zip(passage_sentence_scores, len_sentences) if i]
    sentences = [s for s, i in zip(sentences, len_sentences) if i]
    # sentences = [(i, s) for i, s in zip([i.strip() for i in passage.split('[SEP]')], passage_sentence_scores) if len(i) > 15]
    # if sentences:
    #     passage, passage_sentence_scores = list(map(list, zip(*sentences)))
    # else:
    #     passage = []
    #     passage_sentence_scores = []

    if len(sentences) != len(passage_sentence_scores):
        break

    stuff.append((example.topic, example.docno, sentences, passage_sentence_scores))
# stuff = map(func, tqdm(zip(pred_x.predictions, tokenized_datasets), total=pred_x.predictions.shape[0]))
# pool.close()
# pool.join()




passages = defaultdict(lambda: defaultdict(list))
for topic, docno, passage, score_list in stuff:
    passages[topic][docno].append((passage, score_list))

aa = []
for topic, ps in tqdm((passages.items())):
    for docno, temp in ps.items():
        sentences = [x[0] for x in temp]
        sentence_scores = [x[1] for x in temp]
        aa.append((topic, docno, sum(sentences, []), sum(sentence_scores, [])))

df2 = pd.DataFrame(aa, columns="topic docno passage sentence_scores".split())
df2.to_parquet("bigbird3_passages")
df2 = pd.read_parquet("bigbird3_passages")
df3 = df2.merge(df, on="topic docno".split())
df3['sentences'] = df3.passage.apply(lambda x: [i.strip() for i in x])
df3.passage = df3.sentences.apply(lambda x: ' '.join(x))
df3.to_parquet("data/Top1kBM25.bigbird_passages.snappy.parquet")
df3 = pd.read_parquet("data/Top1kBM25.bigbird_passages.snappy.parquet")
# torch.save(passages, 'tmp_bigbird2')

#%%
