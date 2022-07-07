import json
import os
from collections import Counter

import datasets
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoModelForSequenceClassification, TrainingArguments, \
    default_data_collator, Trainer, TrainerCallback, DataCollatorWithPadding

dataset_name = "pubmed_qa"
datasets_artificial = load_dataset(dataset_name, "pqa_artificial")['train'].train_test_split(0.01)
datasets_labeled = load_dataset(dataset_name, "pqa_labeled")['train'].train_test_split(
    0.5)  # ["train"].train_test_split(0.5)
datasets_unlabeled = load_dataset(dataset_name, "pqa_unlabeled")  # ["train"].train_test_split(0.5)

# %%
MAX_LENGTH = 512  # The maximum length of a feature (question and context)
# MODEL_START_POINT = 'blizrys/biobert-v1.1-finetuned-pubmedqa'
MODEL_START_POINT = 'facebook/muppet-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_START_POINT)
model_name = MODEL_START_POINT.split("/")[-1]


def tokenize_dataset(ds, answer_as_context, pred=None):
    dd = {}
    answer_key = {'yes': 2, 'maybe': 1, 'no': 0} if ds == datasets_labeled else {'yes': 1, 'no': 0}
    for split in ds.keys():
        x = tokenizer(ds[split]["question"],
                      ds[split]["long_answer"]
                      if answer_as_context else
                      [' '.join(x['contexts']) for x in tqdm(ds[split]["context"], desc="fixing contexts...")],
                      truncation="only_second",
                      max_length=MAX_LENGTH,
                      # padding="max_length"
                      )
        d = pd.DataFrame.from_dict({i: x[i] for i in x.keys() if i != 'offset_mapping'})
        if 'final_decision' in ds[split].column_names:
            d['labels'] = ds[split]['final_decision']
            d.labels = d.labels.map(answer_key)
        elif pred is not None:
            d['labels'] = pred.tolist()
        dd[split] = d

    tokenized_datasets = DatasetDict({
        k: Dataset.from_pandas(v) for k, v in dd.items()
    })
    return tokenized_datasets


# %%

def setup(tokenized_datasets, checkpoint, n_epochs, experiment_name):
    output_dir = f"checkpoints/{model_name}-{dataset_name}-{experiment_name}-finetuned"
    if 'labels' in tokenized_datasets['train'].column_names:
        num_labels = Counter(tokenized_datasets['train']['labels']).keys().__len__()
    else:
        num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels, ignore_mismatched_sizes=True)
    batch_size = 32
    args = TrainingArguments(
        output_dir,
        save_strategy="epoch",
        evaluation_strategy="epoch" if "test" in tokenized_datasets.keys() else "no",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=n_epochs,
        weight_decay=0.01,
        push_to_hub=False,
        load_best_model_at_end=True if "test" in tokenized_datasets.keys() else False,
        metric_for_best_model='f1',
    )
    import numpy as np

    if 'labels' in tokenized_datasets['train'].column_names:
        class_weights = Counter(tokenized_datasets['train']['labels'])
        class_weights = (1 - np.array(list(class_weights.values())) / sum(class_weights.values())).astype('float')
    else:
        class_weights = None

    # %%
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        cp = classification_report(labels, preds)
        print(cp)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
        }

    class MyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = torch.tensor(inputs.get("labels")).to(self.model.device)
            outputs = model(**inputs)
            logits = outputs.get("logits")
            if class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().to(self.model.device))
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels)
            return (loss, outputs) if return_outputs else loss

    trainer: Trainer = MyTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"] if "test" in tokenized_datasets.keys() else None,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    class EvaluationCallback(TrainerCallback):
        metrics = []

        def on_evaluate(self, args, state, control, **kwargs):
            print(kwargs['metrics'])
            self.metrics.append(kwargs['metrics'])
            with open(f"{output_dir}/metrics.txt", 'w') as f:
                json.dump(self.metrics, f, indent=4)

    trainer.add_callback(EvaluationCallback())
    return trainer


# %%

print(1)
tokenize_dataset_a_a = tokenize_dataset(datasets_artificial, True)
tokenize_dataset_a_l = tokenize_dataset(datasets_labeled, True)
tokenize_dataset_a_u = tokenize_dataset(datasets_unlabeled, True)
print(2)
tokenize_dataset_c_a = tokenize_dataset(datasets_artificial, False)
print(3)
tokenize_dataset_c_l = tokenize_dataset(datasets_labeled, False)


def train(tokenize_datasets, checkpoint, n_epochs, experiment_name):
    trainer = setup(tokenize_datasets, checkpoint, n_epochs, experiment_name)
    trainer.train()
    trainer.save_model(f"checkpoints/{dataset_name}-{model_name}/{experiment_name}/best")

import os
def predict(tokenized_datasets, checkpoint, n_epochs, experiment_name):
    trainer = setup(tokenized_datasets, checkpoint, n_epochs, experiment_name)
    pred = trainer.predict(tokenized_datasets["train"])
    pred = torch.tensor(pred.predictions).softmax(-1).argmax(-1)
    if not os.path.exists(f'data/{dataset_name}-{model_name}'):
        os.makedirs(f'data/{dataset_name}-{model_name}')
    torch.save(pred, f"data/{dataset_name}-{model_name}/{experiment_name}.pt")

def evaluate(tokenized_datasets, checkpoint, n_epochs, experiment_name):
    trainer = setup(tokenized_datasets, checkpoint, n_epochs, experiment_name)
    trainer.evaluate()

train(tokenize_dataset_a_a, MODEL_START_POINT, 5, 'phase-ii-1')
train(tokenize_dataset_a_l, f"checkpoints/{dataset_name}-{model_name}/phase-ii-1/best", 40, 'phase-ii-2')
predict(tokenize_dataset_a_u, f"checkpoints/{dataset_name}-{model_name}/phase-ii-2/best", 5, 'pred-phase-ii-2')

pred_u = torch.load(f"data/{dataset_name}-{model_name}/pred-phase-ii-2.pt")
tokenize_dataset_c_u = tokenize_dataset(datasets_unlabeled, False,
                                        pred_u)

train(tokenize_dataset_c_a, MODEL_START_POINT, 5, 'pubmedqa-phase-i')
train(tokenize_dataset_c_u, f"checkpoints/{dataset_name}-{model_name}/pubmedqa-phase-i/best", 5, 'pubmedqa-phase-ii')
train(tokenize_dataset_c_l, f"checkpoints/{dataset_name}-{model_name}/pubmedqa-phase-ii/best", 20, 'pubmedqa-phase-iii')

# %%
evaluate(tokenize_dataset_c_l, f"checkpoints/{dataset_name}-{model_name}/pubmedqa-phase-iii/best", 1, "_")