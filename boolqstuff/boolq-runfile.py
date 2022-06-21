from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
# from datasets import load_dataset
import torch
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from boolqstuff.BaseModules import prep_boolq_dataset, NO, YES, IRRELEVANT
from boolqstuff.bert_modules import BoolQBertModule

from boolqstuff.t5_modules import MyLightningDataModule, MyDataset


from utils.util import url2host, url2domain
#%%
if __name__ == '__main__':
    # %%
    # parser = ArgumentParser()
    # # add PROGRAM level args
    # # parser.add_argument("--conda_env", type=str, default="some_name")
    # # parser.add_argument("--notification_email", type=str, default="will@email.com")
    # # add model specific args
    # # parser = MyLightningModel.add_model_specific_args(parser)
    # # parser = pl.Trainer.add_argparse_args(parser)
    # parser.add_argument("--batch_size", default=4, type=int)
    # parser.add_argument("--max_epochs", default=1, type=int)
    # parser.add_argument("--lr", default=1e-5, type=float)
    # parser.add_argument("--t_name", default="microsoft/deberta-base", type=str)
    # parser.add_argument("--load_from", default=None, type=str)
    # parser.add_argument('--augment', action='store_true')
    # parser.add_argument('--infer_all', action='store_true')
    # parser.add_argument('--no_train', action='store_true')
    # parser.add_argument('--no-augment', action='store_false')
    # parser.set_defaults(augment=False)
    # # parser.add_argument("--load_from", default="checkpoints/boolqstuff-simple/deberta-base-num_class=3-lr=1e-05-batch_size=16/epoch=03-valid_F1Score=0.906-valid_Accuracy=0.906.ckpt", type=str)
    # # parser.add_argument("--transformer-type", default="t5", type=str)
    # parser.add_argument("--load_epoch", default=None, type=int)
    # args = parser.parse_known_args()
    # # YES = "▁5.0"
    # # NO = "▁1.0"
    # # IRRELEVANT = "▁3.0"
    # MODEL_NAME = args[0].t_name
    # LR = args[0].lr
    # NUM_CLASSES = 3
    # BATCH_SIZE = args[0].batch_size
    # AUGMENT = args[0].augment
    # LOAD_EPOCH = args[0].load_epoch
    # INFER_ALL = args[0].infer_all
    # NO_TRAIN = args[0].no_train
    # from boolqstuff.BaseModules import prep_boolq_dataset, NO, YES

    # %%
    df = pd.concat([pd.read_parquet("mdt5/output_Top1kBM25_2019_mt5_2019_base-med_with_text").\
                   rename(columns={"docid": "docno", "rang":"ranking"}).\
                       merge(
        pd.read_csv("./data/topics_fixed_extended.tsv.txt", sep="\t")["topic description efficacy".split()],
        on="topic",
        how="inner"
    ),
                    pd.read_parquet("mdt5/output_Top1kBM25_2021_mt5_2021_base-med_with_text")
                        .rename(columns={"docid": "docno","rang":"ranking"}).\
                       merge(
        pd.read_csv("./data/topics_fixed_extended.tsv.txt", sep="\t")["topic description efficacy".split()],
        on="topic",
        how="inner"
    ),
                    pd.read_parquet("data/RunBM25.1k.passages_6_3.top_mt5/")])
    # df = pd.read_parquet("data/RunBM25.1k.passages_6_3.top_mt5/")
    # df = df.rename(columns={"docid": "docno"})
    # df = df.merge(pd.read_csv("./data/RW.txt", sep="\t")["topic description".split()], on="topic", how="inner")



    #%%
    df["source_text"] = df.apply(lambda row: f"{row.description} [SEP] {row.passage}", axis=1)
    dataLoader = DataLoader(df, shuffle=False, batch_size=4)
    # df = df.rename(columns={"stance": "credibility", "credibility": "stance"})
    # df = df[df.stance.ge(0)]


    def prep_sentence(q, p):
        return f"{q} [SEP] {p}"


    # def gen_labels(row):
    #     if row.stance == 2 or row.stance == 0:
    #         return 1
    #     if row.stance == 3:
    #         return 2
    #     if row.stance == 1:
    #         return 0
    #     return float('nan')
    #
    #
    df["target_class"] = 1
    df["target_text"] = IRRELEVANT.replace("▁", "")
    # df_train, df_test = train_test_split(df, test_size=0.2, stratify=df.target_class, random_state=42)

    # %%


    # %%
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    dataset = MyDataset(
        df,
        tokenizer=tokenizer,
        source_max_token_len=512,
        target_max_token_len=2,
    )
    dataLoader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=3).to(0)
    # tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-TinyBERT-L6-v2")
    # model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/paraphrase-TinyBERT-L6-v2",
    #                                                            num_labels=num_classes).to(0)
    lightning_module = BoolQBertModule(
        tokenizer=tokenizer,
        model=model,
        save_only_last_epoch=True,
        labels_text=[NO, IRRELEVANT, YES],
        num_classes=3,
        train_metrics="Accuracy".split(),
        valid_metrics="Accuracy F1Score".split(),
    )

    lightning_module.load_from_checkpoint("./checkpoints/boolqstuff-qrel/deberta-base-lr=1e-05-batch_size=16-aug-noirrel-alltrain/epoch=04.ckpt", tokenizer=tokenizer, model=model)
    lightning_module.to('cuda')
    logits = []
    for stuff in tqdm(dataLoader):
        lightning_module.eval()
        with torch.no_grad():
            output = lightning_module.forward_v(
                input_ids=stuff['source_text_input_ids'].to(lightning_module.device),
                attention_mask=stuff['source_text_attention_mask'].to(lightning_module.device))
            logits.append(output.logits.cpu())

    all_logits = torch.vstack(logits)
    out_df = pd.DataFrame({
        'topic': df.topic,
        'docno': df.docno,
        'mt5': df.score,
        'logits': all_logits.tolist()
    })
    out_df.to_parquet("./mf/RW_passage_6_3.boolq_logits.parquet")

    # %%
    out_df = pd.read_parquet(["./mf/2019_passage_6_3.boolq_logits.parquet",
                              "./mf/2021_passage_6_3.boolq_logits.parquet",
                              "./mf/RW_passage_6_3.mt5_top.boolq_logits.parquet"])
    df = df.merge(out_df, on="topic docno".split(), how="inner")
    df["prob_pos"] = df.logits.apply(lambda x: torch.tensor(x).softmax(-1)[2].item())
    df["prob_neg"] = df.logits.apply(lambda x: torch.tensor(x).softmax(-1)[0].item())

    df["host"] = df.url.apply(url2host)
    df["domain"] = df.url.apply(url2domain)
    # run = pd.concat([pd.read_parquet("mdt5/output_Top1kBM25_2019_mt5_2019_base-med_with_text"),
    #                  pd.read_parquet("mdt5/output_Top1kBM25_2021_mt5_2021_base-med_with_text")])
    # run = run.rename(columns={"docid": "docno"})
    # df = df.merge(run["topic docno score".split()], on="topic docno".split(), how="inner")

    # x = df.groupby("topic host".split()).apply(lambda x: pd.Series([x.prob_neg.mean(), x.prob_pos.mean(), x.description.max(), x.efficacy.max()], index="prob_neg prob_pos description efficacy".split())).reset_index()

    # df = df.groupby("topic").apply(lambda x: x.sort_values("score", ascending=False).head(100)).reset_index(drop=True)
    x = df.groupby("topic host".split()).apply(lambda x: x.nlargest(1, 'score').head(1)).reset_index()

    x = x[['docno', 'score', 'passage', 'url',
       'description', 'efficacy', 'timestamp', 'bm25', 'passage_index',
       'logits', 'prob_pos', 'prob_neg', 'domain']]

    x = x.reset_index()

    # x = df.groupby("topic host".split()).apply(lambda x: pd.Series([x.prob_neg.max(), x.prob_pos.max()], index="neg pos".split())).reset_index()
    x.to_parquet("./mf/run.passage_6_3.boolq_logits.host_max.parquet")





#