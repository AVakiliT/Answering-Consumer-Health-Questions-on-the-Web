import json
from collections import Counter

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from transformers import default_data_collator, TrainingArguments, Trainer, AutoModelForSequenceClassification, \
    AutoTokenizer, TrainerCallback

from github.EncT5.enc_t5 import EncT5ForSequenceClassification, EncT5Tokenizer

dataset_name = 'boolq'
experiment_name = 'binary-classifier'
datasets = load_dataset(dataset_name)

#%%
model_checkpoint = 'facebook/muppet-roberta-base'
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Resize embedding size as we added bos token
if model.config.vocab_size < len(tokenizer.get_vocab()):
    model.resize_token_embeddings(len(tokenizer.get_vocab()))

#%%
max_length = 384
def tokenize_dataset(examples):
    # examples["question"] = [q.lstrip() for q in examples["question"]]
    tokenized_examples = tokenizer(
        examples["question"],
        examples["passage"],
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
    )
    tokenized_examples["labels"] = list(map(int,examples['answer']))
    return tokenized_examples


tokenized_datasets = datasets.map(tokenize_dataset, batched=True, batch_size=1024)
#%%
batch_size = 8
model_name = model_checkpoint.split("/")[-1]
output_dir = f"checkpoints/{model_name}-{dataset_name}-{experiment_name}-finetuned"
args = TrainingArguments(
    output_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)
class_weights = Counter(datasets['train']['answer'])
class_weights = [class_weights[1] / sum(class_weights.values()) , class_weights[0] / sum(class_weights.values())]


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
        labels = torch.tensor(inputs.get("labels")).to(model.device)
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels)
        return (loss, outputs) if return_outputs else loss



trainer = MyTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

metrics = []
class EvaluationCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        metrics.append(kwargs['metrics'])
        with open(f"{output_dir}/metrics.txt", 'w') as f:
            json.dump(metrics, f, indent=4)
trainer.add_callback(EvaluationCallback())


# %%
trainer.evaluate()