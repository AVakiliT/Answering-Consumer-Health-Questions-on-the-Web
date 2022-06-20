from collections import Counter

import torch
from datasets import load_dataset, load_metric, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoModelForSequenceClassification, TrainingArguments, \
    default_data_collator, Trainer

dataset = load_dataset("pubmed_qa", "pqa_artificial")
dataset = dataset['train'].train_test_split(test_size=0.1)
# dataset = DatasetDict({
#     'train': load_dataset("pubmed_qa", "pqa_artificial")['train'],
#     'text': load_dataset("pubmed_qa", "pqa_labeled")['train']
# })
# metric = load_metric('pubmed_qa')
# %%
max_length = 512  # The maximum length of a feature (question and context)
doc_stride = 128  # The authorized overlap between two part of the context when splitting it is needed.
model_checkpoint = "blizrys/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-pubmedqa"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

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

    tokenized_examples["label"] = [answer_key[x] for x in examples["final_decision"]]

    return tokenized_examples


tokenized_datasets = dataset.map(prepare_features, batched=True, remove_columns=dataset["train"].column_names)

# %%
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)
model_name = model_checkpoint.split("/")[-1]
batch_size = 8
args = TrainingArguments(
    f"pubmedqa-{model_name}-finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)

# %%
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


data_collator = default_data_collator

c = Counter(tokenized_datasets['train']['label'])
class_weights = 1 / torch.tensor([c.get(0),c.get(1,13000), c.get(2)])
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


trainer = MyTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)



# %%
trainer.train()
# %%

