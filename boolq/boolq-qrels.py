from argparse import ArgumentParser

import pandas as pd
# from datasets import load_dataset
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from boolq.bert_modules import BoolQBertModule
#%%
from boolq.t5_modules import MyLightningDataModule

parser = ArgumentParser()
# add PROGRAM level args
# parser.add_argument("--conda_env", type=str, default="some_name")
# parser.add_argument("--notification_email", type=str, default="will@email.com")
# add model specific args
# parser = MyLightningModel.add_model_specific_args(parser)
# parser = pl.Trainer.add_argparse_args(parser)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--max_epochs", default=2, type=int)
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--t_name", default="microsoft/deberta-base", type=str)
parser.add_argument("--load_from", default=None, type=str)
# parser.add_argument("--load_from", default="checkpoints/boolq-simple/deberta-base-num_class=3-lr=1e-05-batch_size=16/epoch=03-valid_F1=0.906-valid_Accuracy=0.906.ckpt", type=str)
# parser.add_argument("--transformer-type", default="t5", type=str)
args = parser.parse_known_args()
# YES = "▁5.0"
# NO = "▁1.0"
# IRRELEVANT = "▁3.0"
MODEL_NAME = args[0].t_name
LR = args[0].lr
NUM_CLASSES = 3
BATCH_SIZE = args[0].batch_size
LOAD_FROM = args[0].load_from
LOAD_CHECKPOINT_PATH = LOAD_FROM.split('/')[-3] + '-' + LOAD_FROM.split('/')[-2] + '-' + LOAD_FROM.split('/')[-1].split('-')[0] + '/' if LOAD_FROM else ''
YES = "▁yes"
NO = "▁no"
IRRELEVANT = "▁irrelevant"
#%%
df = pd.read_parquet("./qreldataset/2019_mt5_dataset.parquet")

df["source_text"] = df.apply(lambda row : f"{row.description} [SEP] {row.text}", axis=1)
df = df.rename(columns={"stance": "credibility", "credibility": "effective"})
df = df[df.effective != -2]
df = df[df.effective != 2]

def prep_sentence(q, p):
    return f"{q} [SEP] {p}"

def gen_labels(row):
    if row.usefulness == 0 or row.effective == 0:
        return 0
    if row.effective == 3:
        return 2
    if row.effective == 1:
        return 1

df["target_class"] = df.apply(gen_labels, axis=1)
df["target_text"] = df.target_class.map({0: NO.replace("▁", ""), 1: IRRELEVANT.replace("▁", ""), 2: YES.replace("▁", "")})
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df.target_class, random_state=42)


weights = torch.tensor((1 / (df_train.target_class.value_counts() / df_train.shape[0]).sort_index()).to_list())
weights = weights / weights.sum()

#%%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES).to(0)
# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-TinyBERT-L6-v2")
# model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/paraphrase-TinyBERT-L6-v2",
#                                                            num_labels=num_classes).to(0)
lightning_module = BoolQBertModule(
    tokenizer=tokenizer,
    model=model,
    save_only_last_epoch=True,
    labels_text=[NO, IRRELEVANT, YES],
    num_classes=NUM_CLASSES,
    train_metrics="Accuracy".split(),
    valid_metrics="Accuracy F1".split(),
    weights=weights,
    lr=LR
)

#%%
source_max_token_len = 512
target_max_token_len = 2
data_module = MyLightningDataModule(
    df_train,
    df_test,
    tokenizer=tokenizer,
    batch_size=BATCH_SIZE,
    source_max_token_len=source_max_token_len,
    target_max_token_len=target_max_token_len,
    num_workers=1
)
callbacks = [TQDMProgressBar(refresh_rate=1)]

#         # add gpu support
gpus = 1
#
#         # add logger
loggers = True
#
CHECKPOINT_PATH = f"checkpoints/boolq-qrel/{LOAD_CHECKPOINT_PATH}{MODEL_NAME.split('/')[-1]}-lr={args[0].lr}-batch_size={BATCH_SIZE}"
precision = 32
MAX_EPOCHS = args[0].max_epochs
checkpoint_callback = ModelCheckpoint(
    monitor="valid_F1",
    filename="{epoch:02d}-{valid_F1:.3f}-{valid_Accuracy:.3f}",
    mode="max",
    dirpath=CHECKPOINT_PATH,
    every_n_epochs=1,
    save_top_k=2
)
callbacks.append(checkpoint_callback)

trainer = Trainer(
    logger=loggers,
    callbacks=callbacks,
    max_epochs=MAX_EPOCHS,
    gpus=gpus,
    precision=precision,
    log_every_n_steps=1,
    default_root_dir="checkpoints",
    enable_checkpointing=True,
)

if LOAD_FROM :
    lightning_module.load_from_checkpoint(
        LOAD_FROM,
        tokenizer=tokenizer,
        model=model,
        save_only_last_epoch=True,
        num_classes=NUM_CLASSES,
        labels_text=[NO, IRRELEVANT, YES],
        train_metrics="Accuracy".split(),
        valid_metrics="Accuracy F1".split(),
        weights=weights
    )

trainer.fit(lightning_module, data_module, ckpt_path=None)