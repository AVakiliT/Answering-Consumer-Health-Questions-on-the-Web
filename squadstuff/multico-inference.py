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

window_size, step = 1, 1
# df = pd.read_parquet(f"/project/6004803/avakilit/Trec21_Data/data/RunBM25.1k.passages_{window_size}_{step}")

df = pd.concat([
    pd.read_parquet("/project/6004803/avakilit/Trec21_Data/data/Top1kBM25_2019"),
    pd.read_parquet("/project/6004803/avakilit/Trec21_Data/data/Top1kBM25"),
pd.read_parquet("/project/6004803/avakilit/Trec21_Data/Top1kRWBM25_32p")]).reset_index()

topics = pd.read_csv("data/topics_fixed_extended.tsv.txt", sep='\t')


df2 = df["topic docno url text score".split()].merge(topics["topic description".split()], on="topic", how="inner")
dataset = Dataset.from_pandas(df2)


# %%
max_length = 2048  # The maximum length of a feature (question and context)
doc_stride = 128  # The authorized overlap between two part of the context when splitting it is needed.

# model_checkpoint = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
# model_checkpoint = 'microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL'
# model_checkpoint = 'microsoft/deberta-base'
# model_checkpoint = 'l-yohai/bigbird-roberta-base-mnli'
model_checkpoint = 'google/bigbird-roberta-base'
model_name = model_checkpoint.split("/")[-1]
model_checkpoint = f"checkpoints/{model_name}-mash-qa-tokenclassifier-binary-finetuned/best"
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
        examples["text"],
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
    )
    return tokenized_examples


tokenized_datasets = dataset.map(f, batched=True, batch_size=32)
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
    f"checkpoints/{model_name}-mash-qa-tokenclassifier-binary-finetuned",

    learning_rate=2e-5,
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
            weight=torch.tensor(class_weights).to(model.device)
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


trainer: Trainer = MyTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,

)


class EvaluationCallback(TrainerCallback):
    metrics = []
    def on_evaluate(self, args, state, control, **kwargs):
        try:
            with open(f"checkpoints/{model_name}-mash-qa-tokenclassifier-binary-finetuned/metrics.txt", 'r') as f:
                self.metrics = json.load(f)
        except Exception:
            self.metrics = []
        self.metrics.append(kwargs['metrics'])
        with open(f"checkpoints/{model_name}-mash-qa-tokenclassifier-binary-finetuned/metrics.txt", 'w') as f:
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
trainer.evaluate(tokenized_datasets['test'])
p = trainer.predict(tokenized_datasets['test'])