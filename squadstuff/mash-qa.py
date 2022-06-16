# dataset question context binary

import json
from pathlib import Path
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from torch import nn
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, \
    default_data_collator, AutoModelForSequenceClassification

# %%
dfs = []
splits = "train val test".split()
columns = "question sentence label".split()
for split in splits:
    with open(f"./data/mashqa_data/{split}_webmd_squad_v2_full.json") as f:
        s = json.load(f)

    stuff = []
    for i in s['data']:
        sent_list = i['paragraphs'][0]['sent_list']
        for question in i['paragraphs'][0]['qas']:
            q = question['question']
            answer_aspans = question['answers'][0]['answer_span']
            for sent_number, sentence in enumerate(sent_list):
                stuff.append((q, sentence, 1 if sent_number in answer_aspans else 0))
    dfs.append(pd.DataFrame(stuff, columns=columns))

dfs[0] = dfs[0].drop(dfs[0][dfs[0].label.eq(0)].sample(frac=.8).index)

datasets = DatasetDict({split: ds for split, ds in zip(splits, [Dataset.from_pandas(df) for df in dfs])})

# %%
model_checkpoint = "microsoft/deberta-base"
max_length = 384  # The maximum length of a feature (question and context)
doc_stride = 128  # The authorized overlap between two part of the context when splitting it is needed.
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# %%
def toeknize_dataset(examples):
    examples["question"] = [q.lstrip() for q in examples["question"]]
    tokenized_examples = tokenizer(
        examples["question"],
        examples["sentence"],
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
    )
    tokenized_examples["label"] = examples["label"]
    return tokenized_examples


tokenized_datasets = datasets.map(toeknize_dataset, batched=True, batch_size=1024)
# %%
# Path(disk_path).mkdir(parents=True, exist_ok=True)
#
# tokenized_datasets.save_to_disk('data/mahqa_classification_tokenized')
# %%
# tokenized_datasets = DatasetDict.load_from_disk('data/mahqa_classification_tokenized')
# %%
batch_size = 8
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)
class_weights = (dfs[0].label.value_counts().sum() / dfs[0].label.value_counts()).to_list()
criterion = torch.nn.BC(weights=class_weights)


# %%
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


data_collator = default_data_collator


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


trainer = MyTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# %%
trainer.train()
