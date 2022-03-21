from argparse import ArgumentParser
from typing import Optional, Any

import numpy as np
import torch
import pandas as pd
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, STEP_OUTPUT
from tldextract import tldextract
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration

from boolq.BaseModules import ClassifierLightningModel
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
parser = ArgumentParser()
parser.add_argument("--boolq_checkpoint", default="checkpoints/lightning_logs/version_47/checkpoints/bert-epoch=00-valid_F1=0.668-valid_Accuracy=0.668.ckpt", type=str)
# parser.add_argument("--transformer-type", default="t5", type=str)
args = parser.parse_known_args()





# %%
from boolq.bert_modules import BoolQBertModule
from pipeline.pipeline_modules import PipelineDataModule, PipelineModule


def prep_sentence(row):
    return f"{row.description} [SEP] {row.passage}"



#%%
if __name__ == '__main__':
#%%
    train_dir = "mdt5/output_Top1kBM25_2019_mt5_2019_base-med_with_text/"
    val_dir = "mdt5/output_Top1kBM25_2021_mt5_2021_base-med_with_text/"

    topics_df = pd.read_csv("./data/topics.csv", sep="\t")
    topics_df = topics_df[topics_df.efficacy != 0]
    topics_df.efficacy = topics_df.efficacy.map({-1: 0, 1: 1})

    # def get_df(folder):
    df = pd.read_parquet(train_dir)
    domains = df.url.apply(lambda x: '.'.join(tldextract.extract(x)[-2:]))
    df["domain"] = domains

    # df = df[df.rang <= 20]
    # xx = df.groupby("topic").agg({"passage": list, "domain": list})


    df = pd.merge(df, topics_df["topic query description efficacy".split()], how="inner", on="topic")
    df["source_text"] = df.apply(prep_sentence, axis=1)



    # df_train = get_df(train_dir)
    # df_val = get_df(val_dir)

    emb_weight = torch.load("./weights/domain_emb_graphsage.pt")['weight']
    emb = nn.Embedding.from_pretrained(emb_weight)
    domain2id = torch.load("./weights/domain_emb_graphsage_w2i.pt")
    df["domain_id"] = df.domain.map(domain2id)
    df = df.dropna()
    df["domain_id"] = df.domain_id.astype(int)
    train_df = df.groupby("topic efficacy".split()).head(3).groupby("topic efficacy".split()).agg({"source_text": list, "domain_id": list}).reset_index()


    pretrained_model = "microsoft/deberta-base"
    # pretrained_model = "bert-base-uncased"
    # pretrained_model = "t5-base"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base")
    qa=BoolQBertModule.load_from_checkpoint(args[0].boolq_checkpoint, model=model, tokenizer=tokenizer)
    data_module = PipelineDataModule(
        train_df=train_df,
        val_df=train_df,
        batch_size=1,
        source_max_token_len=512,
        tokenizer=tokenizer,

    )

    data_module.setup()


    lightning_module = PipelineModule(
        model=model,tokenizer=tokenizer,emb=emb,
        weight=None,train_metrics="Accuracy".split(), val_metrics="Accuracy F1".split(),
        lr=1e-5,num_classes=1
    )

    #%%
    max_epochs =1
    precision = 32
    callbacks = [TQDMProgressBar(refresh_rate=1)]

    gpus = 0

    loggers = True

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_F1",
        filename=f"{'lol'}-" + "{epoch:02d}-{valid_F1:.3f}-{valid_Accuracy:.3f}",
        mode="max",
        every_n_epochs=1,
        save_top_k=2
    )
    callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_epochs=max_epochs,
        gpus=gpus,
        precision=precision,
        log_every_n_steps=1,
        default_root_dir="checkpoints",
        enable_checkpointing=True,
    )
    #
    # # fit trainer
    # lightning_module.fix_stupid_metric_device_bs()
    trainer.fit(lightning_module, data_module)



