# dataset question context binary
import re
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
from pathlib import Path
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from torch import nn
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, \
    default_data_collator, AutoModelForSequenceClassification, DataCollatorForTokenClassification, \
    AutoModelForTokenClassification, TrainerCallback, IntervalStrategy, BigBirdForTokenClassification

#%%

# %%
from squadstuff.pegasus import BigBirdPegasusForTokenClassification

max_length = 2048  # The maximum length of a feature (question and context)
doc_stride = 128  # The authorized overlap between two part of the context when splitting it is needed.

# model_checkpoint = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
# model_checkpoint = 'microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL'
# model_checkpoint = 'microsoft/deberta-base'
# model_checkpoint = 'l-yohai/bigbird-roberta-base-mnli'
model_checkpoint = 'google/bigbird-roberta-base'
model_name = model_checkpoint.split("/")[-1]
out_dir = f"checkpoints/{model_name}-mash-qa-tokenclassifier-binary-sep-finetuned-2"
# model_checkpoint = 'distilbert-base-uncased'
# model_checkpoint = 'google/bigbird-pegasus-large-pubmed'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=2, ignore_mismatched_sizes=True)
model2 = BigBirdForTokenClassification.from_pretrained(f"{out_dir}/best", num_labels=2, ignore_mismatched_sizes=True)


# model_checkpoint = 'google/bigbird-pegasus-large-pubmed'
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# model = BigBirdPegasusForTokenClassification.from_pretrained(model_checkpoint, num_labels=2, ignore_mismatched_sizes=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#%%
from github.EncT5.enc_t5 import EncT5ForSequenceClassification, EncT5Tokenizer
def get_labels(row):
    ls = [-100] * len(row.input_ids)
    for l, i in zip(row.sentence_labels, np.where(np.array(row.input_ids) == 66)[0][1:]):
        ls[i] = l
    return ls

dfs = []
splits = "train val test".split()
columns = "question sentences sentence_labels".split()
for split in splits:
    with open(f"./data/mashqa_data/{split}_webmd_squad_v2_full.json") as f:
        s = json.load(f)

    stuff = []
    for i in s['data']:
        sent_list = i['paragraphs'][0]['sent_list']
        for question in i['paragraphs'][0]['qas']:
            query = question['question']
            answer_aspans = question['answers'][0]['answer_span']
            span_labels = [1 if sent_number in answer_aspans else 0 for sent_number in range(len(sent_list))]
            stuff.append((query, sent_list, span_labels))
    df = pd.DataFrame(stuff, columns=columns)
    df['source_text'] = df.apply(lambda x: f"[CLS] {x.question} [SEP]  {' [SEP] '.join(x.sentences)}  [SEP]", axis=1)
    x = tokenizer(df.source_text.to_list(), max_length=max_length, add_special_tokens=False, truncation=True)
    # df['source_text'] = df.apply(lambda x: ' [SEP] '.join(x.sentences), axis=1)
    # x = tokenizer(df.question.to_list(),
    #               df.source_text.to_list(),
    #               max_length=max_length,
    #               padding="max_length",
    #               add_special_tokens=False,
    #               truncation="only_second")
    x = pd.DataFrame.from_dict({i: x[i] for i in x.keys() if i != 'offset_mapping'})
    df = pd.concat([df, x], axis=1)
    df['labels'] = df.apply(get_labels, axis=1)
    dfs.append(df)


# for i in range(0, 3):
# dfs[i] = dfs[i].drop(dfs[i].sample(frac=.9).index)
# dfs[i] = dfs[i].drop(dfs[i].sample(frac=.0).index)

# print(dfs[0].label.value_counts())
tokenized_datasets = DatasetDict({split: ds for split, ds in zip(splits, [Dataset.from_pandas(df) for df in dfs])})


# %%
# disk_path = 'data/mahqa_classification_tokenized'
# Path(disk_path).mkdir(parents=True, exist_ok=True)
#
# tokenized_datasets.save_to_disk(disk_path)
# #%%
# tokenized_datasets = DatasetDict.load_from_disk(disk_path)
# %%
batch_size = 2

args = TrainingArguments(
    out_dir,
    learning_rate=2e-5,
    tf32=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=4,
    weight_decay=0.01,
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    save_strategy="epoch",
    evaluation_strategy="epoch",
)
class_weights = dfs[0]['sentence_labels'].apply(Counter).sum()
class_weights = [sum(class_weights.values()) / class_weights[0], sum(class_weights.values()) / class_weights[1]]


# %%
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    l = labels.reshape(-1)[labels.reshape(-1) != -100]
    p = preds.reshape(-1)[labels.reshape(-1) != -100]
    precision, recall, f1, _ = precision_recall_fscore_support(l, p, average='macro')
    acc = accuracy_score(l, p)
    print(classification_report(l, p))
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights).to(self.model.device)
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


trainer: Trainer = MyTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,

)


class EvaluationCallback(TrainerCallback):
    metrics = []
    def on_evaluate(self, args, state, control, **kwargs):
        print(kwargs['metrics'])
        try:
            with open(f"{out_dir}/metrics.txt", 'r') as f:
                self.metrics = json.load(f)
        except Exception:
            self.metrics = []
        self.metrics.append(kwargs['metrics'])
        with open(f"{out_dir}/metrics.txt", 'w') as f:
            json.dump(self.metrics, f, indent=4)


evaluation_callback = EvaluationCallback()
trainer.add_callback(evaluation_callback)
# %%
# trainer.train('BiomedNLP-PubMedBERT-base-uncased-abstract-mash-qa-binary-finetuned/checkpoint-29500')
# trainer.train('t5-large-mash-qa-binary-finetuned/checkpoint-16000')
# trainer.train()
# trainer.train('checkpoints/bigbird-roberta-base-mash-qa-tokenclassifier-binary-finetuned/')
# %%

# trainer.save_model(f"checkpoints/{model_name}-mash-qa-tokenclassifier-binary-finetuned/best",)
# concatenate_datasets([datasets["train"], datasets["validate"], datasets["test"]])

#%%
trainer.train()
trainer.save_model(f"{out_dir}/best2")
# trainer.evaluate(tokenized_datasets['test'])
# p = trainer.predict(tokenized_datasets['test'])

#%%
pred_train = model2.predict(tokenized_datasets["train"])