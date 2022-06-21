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
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from boolqstuff.BaseModules import prep_boolq_dataset, NO, YES, IRRELEVANT
from boolqstuff.bert_modules import BoolQBertModule

from boolqstuff.t5_modules import MyLightningDataModule


if __name__ == '__main__':
    # %%
    parser = ArgumentParser()
    # add PROGRAM level args
    # parser.add_argument("--conda_env", type=str, default="some_name")
    # parser.add_argument("--notification_email", type=str, default="will@email.com")
    # add model specific args
    # parser = MyLightningModel.add_model_specific_args(parser)
    # parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_epochs", default=1, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--t_name", default="microsoft/deberta-base", type=str)
    parser.add_argument("--load_from", default=None, type=str)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--infer_all', action='store_true')
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--no-augment', action='store_false')
    parser.set_defaults(augment=False)
    # parser.add_argument("--load_from", default="checkpoints/boolqstuff-simple/deberta-base-num_class=3-lr=1e-05-batch_size=16/epoch=03-valid_F1Score=0.906-valid_Accuracy=0.906.ckpt", type=str)
    # parser.add_argument("--transformer-type", default="t5", type=str)
    parser.add_argument("--load_epoch", default=None, type=int)
    args = parser.parse_known_args()
    # YES = "▁5.0"
    # NO = "▁1.0"
    # IRRELEVANT = "▁3.0"
    MODEL_NAME = args[0].t_name
    LR = args[0].lr
    NUM_CLASSES = 3
    BATCH_SIZE = args[0].batch_size
    AUGMENT = args[0].augment
    LOAD_EPOCH = args[0].load_epoch
    INFER_ALL = args[0].infer_all
    NO_TRAIN = args[0].no_train
    from boolqstuff.BaseModules import prep_boolq_dataset, NO, YES

    # %%
    df = pd.read_parquet("./qreldataset/2019_mt5_dataset.parquet")

    df["source_text"] = df.apply(lambda row: f"{row.description} [SEP] {row.text}", axis=1)
    df = df.rename(columns={"stance": "credibility", "credibility": "stance"})
    df = df[df.stance.ge(0)]


    def prep_sentence(q, p):
        return f"{q} [SEP] {p}"


    def gen_labels(row):
        if row.stance == 2 or row.stance == 0:
            return 1
        if row.stance == 3:
            return 2
        if row.stance == 1:
            return 0
        return float('nan')


    df["target_class"] = df.apply(gen_labels, axis=1)
    df["target_text"] = df.target_class.map(
        {0: NO.replace("▁", ""), 1: IRRELEVANT.replace("▁", ""), 2: YES.replace("▁", "")})
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df.target_class, random_state=42)

    # %%
    if INFER_ALL:
        df_train =df
        df_test = df

    if AUGMENT:
        def prep_bert_sentence(q, p):
            return f"{q} [SEP] {p}"


        df_train_aug, _ = prep_boolq_dataset(
            prep_sentence=prep_bert_sentence,
            neg_sampling=False)

        df_train = pd.concat([df_train, df_train_aug])


    weights = torch.tensor((1 / (df_train.target_class.value_counts() / df_train.shape[0]).sort_index()).to_list())
    weights = weights / weights.sum()

    # %%
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
        valid_metrics="Accuracy F1Score".split(),
        weights=weights,
        lr=LR
    )

    # %%
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
    # loggers = True
    loggers = TensorBoardLogger(save_dir="logs/")
    #
    CHECKPOINT_PATH = f"checkpoints/boolq-qrel/{MODEL_NAME.split('/')[-1]}-lr={args[0].lr}-batch_size={BATCH_SIZE}{'-aug' if AUGMENT else ''}-noirrel{'-alltrain' if INFER_ALL else ''}"
    precision = 32
    MAX_EPOCHS = args[0].max_epochs
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_F1Score",
        filename="{epoch:02d}",
        mode="max",
        dirpath=CHECKPOINT_PATH,
        every_n_epochs=1,
        save_top_k=-1,

    )


    class CustomCallback(Callback):
        def __init__(self):
            self.val_outs = []
            self.current_epoch = LOAD_EPOCH or -1


        def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                                    outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int,
                                    dataloader_idx: int) -> None:
            self.val_outs.append(outputs)

        def on_validation_epoch_end(self, trainer, pl_module):
            # print(len(self.val_outs))
            prediction = np.vstack([output['logits'] for output in self.val_outs])
            target = np.hstack([output['target'] for output in self.val_outs])
            if self.current_epoch >=0:
                path = Path(CHECKPOINT_PATH + f"/metrics/epoch-{self.current_epoch}.txt")
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w') as f:
                    print(classification_report(target, prediction.argmax(-1), zero_division=1), file=f)
                path = Path(CHECKPOINT_PATH + f"/history/epoch-{self.current_epoch}.pt")
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save((target, prediction), f=path)
            self.current_epoch += 1

    custom_callback = CustomCallback()
    callbacks.extend([checkpoint_callback, custom_callback])

    trainer = Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_epochs=MAX_EPOCHS,
        gpus=gpus,
        precision=precision,
        log_every_n_steps=1,
        default_root_dir="checkpoints",
        enable_checkpointing=True,
        resume_from_checkpoint=CHECKPOINT_PATH + f'/epoch={LOAD_EPOCH:02}.ckpt' if LOAD_EPOCH else None,
    )

    if not NO_TRAIN:
        trainer.fit(lightning_module, data_module)
    else:
        trainer.validate(lightning_module, data_module)
