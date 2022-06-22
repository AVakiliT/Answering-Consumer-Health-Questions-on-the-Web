import json
from collections import Counter

import torch
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoModelForSequenceClassification, TrainingArguments, \
    default_data_collator, Trainer, TrainerCallback

dataset_name = "pubmed_qa"
experiment_name = "phaseI"
datasets_artificial = load_dataset(dataset_name, "pqa_artificial")
datasets_labeled = load_dataset(dataset_name, "pqa_labeled")["train"].train_test_split(0.5)
# datasets_artificial["train"] = concatenate_datasets([datasets_artificial["train"], datasets_labeled["train"]], axis=0)
# datasets_artificial["test"] = datasets_labeled["test"]
# dataset = DatasetDict({
#     'train': load_dataset("pubmed_qa", "pqa_artificial")['train'],
#     'text': load_dataset("pubmed_qa", "pqa_labeled")['train']
# })
# metric = load_metric('pubmed_qa')
# %%
max_length = 512  # The maximum length of a feature (question and context)
doc_stride = 128  # The authorized overlap between two part of the context when splitting it is needed.
# model_checkpoint = 'blizrys/biobert-v1.1-finetuned-pubmedqa'
model_checkpoint = 'facebook/muppet-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

pad_on_right = tokenizer.padding_side == "right"
assert isinstance(tokenizer, PreTrainedTokenizerFast)

answer_key = {'yes': 2, 'maybe': 1, 'no': 0}


def prepare_features(examples):
    question = examples["question"]
    context = [' '.join(x['contexts']) for x in examples["context"]]
    tokenized_examples = tokenizer(
        question if pad_on_right else context,
        context if pad_on_right else context,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        padding="max_length",
    )
    2+2
    tokenized_examples["label"] = [answer_key[x] for x in examples["final_decision"]]

    return tokenized_examples


# tokenized_datasets = datasets_artificial.map(prepare_features, batched=True, batch_size=1024, remove_columns=datasets_artificial["train"].column_names)
# tokenized_datasets2 = datasets_labeled.map(prepare_features, batched=True, remove_columns=datasets_labeled["train"].column_names)

# tokenized_datasets["train"] = concatenate_datasets([tokenized_datasets["train"], tokenized_datasets2["train"]])
# tokenized_datasets['test'] = tokenized_datasets2['test']
tokenized_datasets = datasets_labeled.map(prepare_features, batch_size=1024, batched=True)
# %%
batch_size = 8
model_name = model_checkpoint.split("/")[-1]
output_dir = f"checkpoints/{model_name}-{dataset_name}-{experiment_name}-finetuned"
args = TrainingArguments(
    output_dir,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    # save_steps=2000,
    # eval_steps=2000,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)
class_weights = Counter(tokenized_datasets['train']['label'])
class_weights = [
    (sum(class_weights.values()) - class_weights[0]) / (2 *sum(class_weights.values())),
    (sum(class_weights.values()) - class_weights[1]) / (2 * sum(class_weights.values())),
    (sum(class_weights.values()) - class_weights[2]) / (2 * sum(class_weights.values()))
]


# %%
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    cp = classification_report(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'report': cp,
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



trainer : Trainer = MyTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

metrics = []
class EvaluationCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        print(kwargs['metrics'])
        metrics.append(kwargs['metrics'])
        with open(f"{output_dir}/metrics.txt", 'w') as f:
            json.dump(metrics, f, indent=4)
trainer.add_callback(EvaluationCallback())


# %%
trainer.train()

print(trainer.evaluate()['eval_report'])