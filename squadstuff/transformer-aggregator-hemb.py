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

# special_tokens_dict = {'additional_special_tokens': ['[HON]']}
# num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
# model.resize_token_embeddings(len(tokenizer))

THRESHOLD = 0.75
df = pd.read_parquet(f"data/RunBM25.1k.passages_bigbird2_{int(THRESHOLD*100)}.top_mt5")
df = df.reset_index(drop=True)
MAX_LENGTH = 512

df["host"] = df.url.progress_apply(url2host)
host_cat = df.host.astype("category")
host_mapping = {v: k for k, v in enumerate(host_cat.cat.categories)}
df["host_id"] = df.host.map(lambda x: host_mapping[x]).to_list()

honcode = pd.read_csv("./data/found_HONCode_hosts_no_dups", sep=" ", header=None)[2].to_list()
df["honcode"] = df.host.isin(honcode).astype("float")
df["text_in"] = df.progress_apply(
    # lambda x: f"{x.host} {'[HON]' if x.honcode else ''} {' '.join([sent for sent, score in zip(x.sentences, x.sentence_scores) if score > 0.75])}",
    # lambda x: f"{x.host} {' '.join([sent for sent, score in zip(x.sentences, x.sentence_scores) if score > 0.75])}",
    lambda x: f"{' '.join([sent for sent, score in zip(x.sentences, x.sentence_scores) if score > 0.75])}",
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
ds["host_id"] = df.host_id
ds["honcode"] = df.honcode

MAX_LEN = 512
topics_2019 = list(range(1, 51 + 1))
topics_2021 = list(range(101, 150 + 1))
topics_2022 = list(range(151, 200 + 1))
topics_rw = list(range(1001, 1090 + 1))

dc = DataCollatorWithPadding(tokenizer=tokenizer)
batch_size = 32

model = model.cuda()
bias = nn.parameter.Parameter(torch.zeros(1).squeeze().cuda())
host_Weights = nn.Embedding(num_embeddings=host_cat.cat.codes.max() + 1, embedding_dim=1).cuda()
hon_weight = nn.parameter.Parameter(torch.zeros(1).squeeze().cuda())
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam([
    dict(params=model.parameters(), lr=1e-5),
    dict(params=bias, lr=1e-3),
    dict(params=host_Weights.parameters(), lr=1e-3),
    dict(params=hon_weight, lr=1e-3),
])

from pathlib import Path

out_dir = f'checkpoints/transformer-agg-hemb_RW-train_k={batch_size}'

Path(out_dir).mkdir(parents=True, exist_ok=True)
best_dev_acc = 0

for i_epoch in range(12):
    losses = []
    accs = []
    # if i_epoch == 5:
    #     batch_size = 24
    ds_top_bs = ds.groupby("topic").apply(lambda x: x.head(batch_size)).reset_index(drop=True)
    dl = DataLoader(Dataset.from_pandas(ds_top_bs[ds_top_bs.topic.isin(topics_rw)]),
                    batch_size=batch_size, collate_fn=dc, shuffle=False)
    dl_v = DataLoader(Dataset.from_pandas(ds_top_bs[ds_top_bs.topic.isin(topics_2019)]), batch_size=batch_size,
                      collate_fn=dc, shuffle=False)

    dl_t = DataLoader(Dataset.from_pandas(ds_top_bs[ds_top_bs.topic.isin(topics_2021)]), batch_size=batch_size,
                      collate_fn=dc, shuffle=False)

    pbar = tqdm(dl)
    for batch in pbar:
        assert ((batch["labels"][0] == batch["labels"]).all())
        optimizer.zero_grad()
        batchc = {k: v.cuda() for k, v in batch.items()}
        output = model(batchc['input_ids'], batchc['token_type_ids'], batchc['attention_mask'])
        probs = output.logits[:, 2] - output.logits[:, 0]
        probs = probs * host_Weights(batchc['host_id']).squeeze(-1)
        final = (probs + bias).mean()
        loss = criterion(final, batchc["labels"][0].clamp(min=0))
        loss.backward()
        optimizer.step()
        loss_item = loss.detach().item()
        losses.append(loss_item)
        accs.append(final.detach().cpu().gt(0).eq(batch["labels"][0].gt(0)).item())
        pbar.set_description(f"[EP {i_epoch:02d}][TRN] loss: {np.mean(losses):.03} acc: {np.mean(accs):.03}")

    for n, dl_e in [('DEV', dl_v), ('TST', dl_t)]:
        losses = []
        accs = []
        pbar = tqdm(dl_e)
        for batch in pbar:
            assert ((batch["labels"][0] == batch["labels"]).all())
            with torch.no_grad():
                batchc = {k: v.cuda() for k, v in batch.items()}
                output = model(batchc['input_ids'], batchc['token_type_ids'], batchc['attention_mask'])
                probs = output.logits.mean(0)
                probs = output.logits[:, 2] - output.logits[:, 0]
                probs = probs * host_Weights(batchc['host_id'])
                final = (probs + bias).mean()
                loss = criterion(final, batchc["labels"][0].clamp(min=0))
                loss_item = loss.detach().item()
                losses.append(loss_item)
                accs.append(final.detach().cpu().gt(0).eq(batch["labels"][0].gt(0)).item())
                pbar.set_description(f"[EP {i_epoch:02d}][{n}] loss: {np.mean(losses):.03} acc: {np.mean(accs):.03}")
        if n == 'DEV' and i_epoch >= 7:
            if np.mean(accs) >= best_dev_acc:
                print(f"saving model {np.mean(accs)} >= {best_dev_acc}")
                best_dev_acc = np.mean(accs)
                torch.save(model.state_dict(), f"{out_dir}/best")

model.load_state_dict(torch.load(f"{out_dir}/best"))
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_START_POINT,
#                                                            num_labels=3,
#                                                            # output_hidden_states=True
#                                                            ).cuda()
# dl_a = DataLoader(Dataset.from_pandas(ds_top_bs.drop(columns=["label"])),
#                   batch_size=batch_size, collate_fn=dc, shuffle=False)
# topic_predictions = []
# for batch in tqdm(dl_a):
#     with torch.no_grad():
#         batchc = {k: v.cuda() for k, v in batch.items()}
#         output = model(batchc['input_ids'], batchc['token_type_ids'], batchc['attention_mask'])
# probs = output.logits[:, 2] - output.logits[:, 0]
# probs = probs * host_Weights(batchc['host_id'])
# final = (probs + bias).mean()
#         topic = batchc['topic'][0].item()
#         topic_predictions.append((topic, final.item()))
#
# b = [1 if i > 0 else -1 for x, i in topic_predictions[101:151]]
#
# tokenizer = AutoTokenizer.from_pretrained(MODEL_START_POINT)
# model2 = AutoModelForSequenceClassification.from_pretrained(MODEL_START_POINT,
#                                                            num_labels=3,
#                                                            # output_hidden_states=True
#                                                            )
#
# from squadstuff.boolq_utils import MyTrainer, compute_metrics, EvaluationCallback
#
#
#
# # model2 = model
#
# args = TrainingArguments(
#     "checkpoints/tmp",
#     save_strategy="no",
#     per_device_eval_batch_size=48,
#     push_to_hub=False,
# )
#
# trainer: Trainer = Trainer(
#     model2,
#     args,
#     data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
#     tokenizer=tokenizer,
# )
#
# #
# temp = ds.drop(columns=["label"])
# pred_x = trainer.predict(Dataset.from_pandas(temp))
#
# preds = pred_x.predictions[:, 2] - pred_x.predictions[:, 0]
# #
# # df["prediction"] = preds
# df["prediction"] = preds.clip(max=-1) + preds.clip(min=1)
# df["topic_predictions"] = df.topic.map(dict(topic_predictions))
#
# alpha = .1
#
#
# def adjust(row):
#     return (2 * row.score) / (1 + np.exp( alpha * row.raw_adjustment))
#
#
# df["raw_adjustment"] = (df.topic_predictions * df.prediction)
# df["adjusted_score"] = df.apply(adjust, axis=1)
#
#
#
#
# #%%
# import pandas as pd
# from utils.util import fixdocno
# dfx = df.sort_values("topic adjusted_score".split(), ascending=[True, False])
# dfx["ranking"] = list(range(1,1001)) * dfx.topic.nunique()
# run = dfx.apply(lambda x: f"{x.topic} Q0 {fixdocno(x.docno)} {x.ranking} {x.adjusted_score} WatS-Bigbird2_{int(THRESHOLD*100)}-MT5-Tagg", axis=1)
# run.to_csv(f"runs/WatS-Bigbird2_{int(THRESHOLD*100)}-MT5-TAggHEmb.all", index=False, header=False)
