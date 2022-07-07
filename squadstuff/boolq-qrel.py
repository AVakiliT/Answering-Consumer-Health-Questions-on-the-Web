from collections import Counter

import numpy as np
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
import torch
#%%
from squadstuff.boolq_utils import MyTrainer, compute_metrics, EvaluationCallback

df = pd.read_parquet("./qreldataset/2019_mt5_dataset.parquet").reset_index(drop=True)
df = df.rename(columns={"stance": "credibility", "credibility": "stance"})
df = df[df.stance.ge(0)]
def gen_labels(row):
    if row.stance == 2 or row.stance == 0:
        return 1
    if row.stance == 3:
        return 2
    if row.stance == 1:
        return 0
    return float('nan')
df["labels"] = df.apply(gen_labels, axis=1)
# %%
MAX_LENGTH = 512  # The maximum length of a feature (question and context)
MODEL_START_POINT = 'blizrys/biobert-v1.1-finetuned-pubmedqa'
# MODEL_START_POINT = f"checkpoints/pubmedqa-biobert-v1.1-finetuned-pubmedqa/pubmedqa-phase-iii/best"
tokenizer = AutoTokenizer.from_pretrained(MODEL_START_POINT)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_START_POINT)
model_name = MODEL_START_POINT.split("/")[-1]

x = tokenizer(
    df.description.to_list(),
    df.text.to_list(),
    truncation="only_second",
    max_length=MAX_LENGTH,
)

d = pd.DataFrame.from_dict({i: x[i] for i in x.keys() if i != 'offset_mapping'})
d["labels"] = df["labels"].to_list()

tokenized_datasets = Dataset.from_pandas(d)
class_weights = Counter(tokenized_datasets['labels'])
class_weights = (1 - np.array(list(class_weights.values())) / sum(class_weights.values())).astype('float')
tokenized_datasets = tokenized_datasets.train_test_split(0.1)
output_dir = 'checkpoints/boolq-qrel2'
batch_size = 32
args = TrainingArguments(
    output_dir,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
)



#%%
trainer: Trainer = MyTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    class_weights=class_weights
)

trainer.add_callback(EvaluationCallback(output_dir=output_dir))

#%%
trainer.train()

