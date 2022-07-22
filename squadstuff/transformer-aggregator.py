# take top senteneces from the top n docs
# prepend hostname
# feed into pubmedqa
# mean
# total 241 topics


import numpy as np
from datasets import Dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from torch import tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
import torch
from tqdm import tqdm

from utils.util import url2host

tqdm.pandas()
MODEL_START_POINT = f"checkpoints/pubmed_qa-biobert-v1.1-finetuned-pubmedqa/pubmedqa-phase-iii/best"
tokenizer = AutoTokenizer.from_pretrained(MODEL_START_POINT)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_START_POINT,
                                                           num_labels=3,
                                                           # output_hidden_states=True
                                                           )
df = pd.read_parquet(f"data/RunBM25.1k.passages_bigbird.top_mt5")
df = df.reset_index(drop=True)
MAX_LENGTH = 512


df["host"] = df.url.progress_apply(url2host)
df["text_in"] = df.progress_apply(
    lambda x: f"{x.host} {' '.join([sent for sent, score in zip(x.sentences, x.sentence_scores) if score > 0.75])}",
    axis=1)


ds = []
k = 1000
for i in trange(0, df.shape[0], k):
    x = tokenizer(
        df[i:i + k].description.to_list(),
        df[i:i + k].text_in.to_list(),
        truncation="only_second",
        max_length=MAX_LENGTH,
    )

    d = pd.DataFrame.from_dict({i: x[i] for i in x.keys() if i != 'offset_mapping'})
    ds.append(d)
ds = pd.concat(ds)

ds = ds.reset_index(drop=True)
ds["topic"] = df.topic
ds["label"] = df.efficacy



MAX_LEN = 512
topics_2019 = list(range(1,51 + 1))
topics_2021 = list(range(101,150 + 1))
topics_2022 = list(range(151,200 + 1))
topics_rw = list(range(1001,1090 + 1))

batch_size = 16
dc = DataCollatorWithPadding(tokenizer=tokenizer)
ds_top = ds.groupby("topic").apply(lambda x: x.head(batch_size)).reset_index(drop=True)
dl = DataLoader(Dataset.from_pandas(ds_top[ds_top.topic.isin(topics_rw)]), batch_size=batch_size, collate_fn=dc, shuffle=False)
dl_v = DataLoader(Dataset.from_pandas(ds_top[ds_top.topic.isin(topics_2021)]), batch_size=batch_size, collate_fn=dc, shuffle=False)

model = model.cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(),lr=1e-5)

for i_epoch in range(10):
    losses = []
    accs = []
    pbar = tqdm(dl)
    for batch in pbar:
        assert ((batch["labels"][0] == batch["labels"]).all())
        optimizer.zero_grad()
        batchc = {k: v.cuda() for k, v in batch.items()}
        output = model(batchc['input_ids'], batchc['token_type_ids'], batchc['attention_mask'])
        probs = output.logits.mean(0)
        final = probs[2] - probs[0]
        loss = criterion(final, batchc["labels"][0].clamp(min=0))
        loss.backward()
        optimizer.step()
        loss_item = loss.detach().item()
        losses.append(loss_item)
        accs.append(final.detach().cpu().gt(0).eq(batch["labels"][0].gt(0)).item())
        pbar.set_description(f"[TRAI] loss: {np.mean(losses):.03} acc: {np.mean(accs):.03}")

    losses = []
    accs = []
    pbar = tqdm(dl_v)
    for batch in pbar:
        assert ((batch["labels"][0] == batch["labels"]).all())
        with torch.no_grad():
            batchc = {k: v.cuda() for k, v in batch.items()}
            output = model(batchc['input_ids'], batchc['token_type_ids'], batchc['attention_mask'])
            probs = output.logits.mean(0)
            final = probs[2] - probs[0]
            loss = criterion(final, batchc["labels"][0].clamp(min=0))
            loss_item = loss.detach().item()
            losses.append(loss_item)
            accs.append(final.detach().cpu().gt(0).eq(batch["labels"][0].gt(0)).item())
            pbar.set_description(f"[TST] loss: {np.mean(losses):.03} acc: {np.mean(accs):.03}")


# args = TrainingArguments(
#     "checkpoints/tmp",
#     save_strategy="no",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     push_to_hub=False,
#     load_best_model_at_end=True,
#     metric_for_best_model='f1',
# )
#
# trainer: Trainer = Trainer(
#     model,
#     args,
#     data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
#     tokenizer=tokenizer,
# )
#
# pred = trainer.predict(tokenized_dataset)
# p = pd.DataFrame(tensor(pred.predictions).softmax(-1).tolist(), columns="prob_neg prob_may prob_pos".split())
# df3 = pd.concat([df2.reset_index(drop=True), p.reset_index(drop=True)], axis=1)
# df3.to_parquet("./data/run.passages_bigbird.qapubmed_sep-logits")
# df3 = pd.read_parquet("./data/run.passages_bigbird.qapubmed_sep-logits")
#
# # %%
# # xxx = df3.groupby("topic docno".split()).apply(lambda x: x.loc[x.sentence_scores.idxmax()]["prob_pos prob_may prob_neg".split()])
# # xxx = xxx.reset_index(drop=True)
# # dfx = df.merge(xxx, on='topic docno'.split(), how="left")
# # dfx = dfx.sort_values("topic score".split(), ascending=[True, False])
# # dfx.prob_pos = dfx.prob_pos.fillna(0)
# # dfx.prob_neg = dfx.prob_neg.fillna(0)
# # dfx.prob_may = dfx.prob_may.fillna(1)
#
# # %%
# from tqdm import tqdm
#
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_START_POINT,
#                                                            num_labels=3,
#                                                            output_hidden_states=True
#                                                            )
# dl = trainer.get_eval_dataloader(tokenized_dataset)
#
# model = model.cuda()
# cls_hiddens = []
# preds = []
# with torch.no_grad():
#     for i in tqdm(dl):
#         i = {k: v.cuda() for k, v in i.items()}
#         a = model(**i)
#         cls = a.hidden_states[-1][:, 0, :].cpu()
#         cls_hiddens.append(cls)
#         preds.append(a.logits.softmax(-1).cpu())
# cls_hiddens = torch.vstack(cls_hiddens)
# preds = torch.vstack(preds)
# p = pd.DataFrame(preds, columns="prob_neg prob_may prob_pos".split())
# p["cls"] = [i.numpy() for i in cls_hiddens]
# df3 = pd.concat([df2.reset_index(drop=True), p.reset_index(drop=True)], axis=1)
# df3.to_parquet("./data/run.passages_bigbird.qapubmed_sep-logits2")
# df3 = pd.read_parquet("./data/run.passages_bigbird.qapubmed_sep-logits2")
