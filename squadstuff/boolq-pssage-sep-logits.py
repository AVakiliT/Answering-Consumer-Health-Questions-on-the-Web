import numpy as np
from datasets import Dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from torch import tensor
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
import torch

MODEL_START_POINT = f"checkpoints/pubmed_qa-biobert-v1.1-finetuned-pubmedqa/pubmedqa-phase-iii/best"
tokenizer = AutoTokenizer.from_pretrained(MODEL_START_POINT)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_START_POINT, num_labels=3)
df = pd.read_parquet(f"data/RunBM25.1k.passages_bigbird.top_mt5")
df = df.reset_index(drop=True)
MAX_LENGTH = 512

df2 = df["topic docno description sentences sentence_scores".split()].set_index("topic docno description".split()).apply(pd.Series.explode).reset_index()

df2 = df2.dropna(subset=["sentence_scores"])


ds = []
k = 1000
for i in trange(0, df2.shape[0], k):
    x = tokenizer(
        df2[i:i + k].description.to_list(),
        df2[i:i + k].sentences.to_list(),
        truncation="only_second",
        max_length=MAX_LENGTH,
    )

    d = pd.DataFrame.from_dict({i: x[i] for i in x.keys() if i != 'offset_mapping'})
    ds.append(d)
ds = pd.concat(ds)
tokenized_dataset = Dataset.from_pandas(ds)

batch_size = 32
args = TrainingArguments(
    "checkpoints/tmp",
    save_strategy="no",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
)

trainer: Trainer = Trainer(
    model,
    args,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    tokenizer=tokenizer,
)

pred = trainer.predict(tokenized_dataset)
p = pd.DataFrame(tensor(pred.predictions).softmax(-1).tolist(), columns="prob_neg prob_may prob_pos".split())
df3 = pd.concat([df2.reset_index(drop=True), p.reset_index(drop=True)], axis=1)
df3.to_parquet("./data/run.passages_bigbird.qapubmed_sep-logits")
df3 = pd.read_parquet("./data/run.passages_bigbird.qapubmed_sep-logits")

#%%
xxx = df3.groupby("topic docno".split()).apply(lambda x: x.loc[x.sentence_scores.idxmax()]["prob_pos prob_may prob_neg".split()])
xxx = xxx.reset_index(drop=True)
dfx = df.merge(xxx, on='topic docno'.split(), how="left")
dfx = dfx.sort_values("topic score".split(), ascending=[True, False])
dfx.prob_pos = dfx.prob_pos.fillna(0)
dfx.prob_neg = dfx.prob_neg.fillna(0)
dfx.prob_may = dfx.prob_may.fillna(1)