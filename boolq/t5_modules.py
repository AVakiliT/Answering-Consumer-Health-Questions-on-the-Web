from typing import List

import torch
import numpy as np
import pandas as pd
import torchmetrics
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import (
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    ByT5Tokenizer,
    PreTrainedTokenizer,
    T5TokenizerFast as T5Tokenizer,
    MT5TokenizerFast as MT5Tokenizer,
)
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelWithLMHead, AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch.nn.functional as F

torch.cuda.empty_cache()
pl.seed_everything(42)


class MyDataset(Dataset):
    """  PyTorch Dataset class  """

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: PreTrainedTokenizer,
            source_max_token_len: int = 512,
            target_max_token_len: int = 512,
    ):
        """
        initiates a PyTorch Dataset Module for input data
        Args:
            data (pd.DataFrame): input pandas dataframe. Dataframe must have 2 column --> "source_text" and "target_text"
            tokenizer (PreTrainedTokenizer): a PreTrainedTokenizer (T5Tokenizer, MT5Tokenizer, or ByT5Tokenizer)
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        """ returns length of data """
        return len(self.data)

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5/MT5 model"""

        data_row = self.data.iloc[index]
        source_text_encoding = self.tokenizer(
            data_row["source_text"],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        target_text_encoding = self.tokenizer(
            data_row["target_text"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = target_text_encoding["input_ids"]
        labels[
            labels == 0
            ] = -100  # to make sure we have correct labels for T5 text generation


        return dict(
            source_text_input_ids=source_text_encoding["input_ids"].flatten(),
            source_text_attention_mask=source_text_encoding["attention_mask"].flatten(),
            labels=labels,
            labels_attention_mask=target_text_encoding["attention_mask"].flatten(),
            target_class=data_row["target_class"].flatten()
        )


class MyLightningDataModule(pl.LightningDataModule):
    """ PyTorch Lightning data class """

    def __init__(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            tokenizer: PreTrainedTokenizer,
            batch_size: int = 4,
            source_max_token_len: int = 512,
            target_max_token_len: int = 512,
            num_workers: int = 2,
    ):
        """
        initiates a PyTorch Lightning Data Module
        Args:
            train_df (pd.DataFrame): training dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            test_df (pd.DataFrame): validation dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            tokenizer (PreTrainedTokenizer): PreTrainedTokenizer (T5Tokenizer, MT5Tokenizer, or ByT5Tokenizer)
            batch_size (int, optional): batch size. Defaults to 4.
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        super().__init__()

        self.train_dataset = None
        self.val_dataset = None
        self.train_df = train_df
        self.val_df = val_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = MyDataset(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )
        self.val_dataset = MyDataset(
            self.val_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )

    def train_dataloader(self):
        """ training dataloader """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """ validation dataloader """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class MyLightningModel(pl.LightningModule):
    """ PyTorch Lightning Model class"""

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def __init__(
            self,
            tokenizer,
            model,
            train_metrics="",
            valid_metrics="",
            weights=None,
            outputdir: str = "outputs",
            save_only_last_epoch: bool = False,
            num_classes=2
            , labels_text=None,
    lr=1e-5):
        """
        initiates a PyTorch Lightning Model
        Args:
            tokenizer : T5/MT5/ByT5 tokenizer
            model : T5/MT5/ByT5 model
            outputdir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
            save_only_last_epoch (bool, optional): If True, save just the last epoch else models are saved for every epoch
        """
        super().__init__()
        self.save_hyperparameters("num_classes", "lr", "num_classes")
        self.weights = weights
        for m in train_metrics:
            setattr(self, "train_" + m, getattr(torchmetrics, m)(num_classes=num_classes))
            # getattr(self, "train_" + m).device = "cpu"
        self.train_metrics = train_metrics
        for m in valid_metrics:
            setattr(self, "valid_" + m, getattr(torchmetrics, m)(num_classes=num_classes))
            # getattr(self, "valid_" + m).device = "cpu"
        self.valid_metrics = valid_metrics
        self.model = model
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.outputdir = outputdir
        self.average_training_loss = None
        self.average_validation_loss = None
        self.num_classes = num_classes
        self.save_only_last_epoch = save_only_last_epoch
        self.max_len = 3
        self.lr = lr
        # self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes)
        # self.valid_acc = torchmetrics.Accuracy(num_classes=self.num_classes)
        # self.valid_auroc = torchmetrics.AUROC(num_classes=self.num_classes)
        self.train_loss = torchmetrics.MeanMetric()
        self.valid_loss = torchmetrics.MeanMetric()
        if labels_text:
            if num_classes == 2:
                self.label_token_mapping = np.array(self.tokenizer.convert_tokens_to_ids(labels_text))[[0, 2]]
            else:
                self.label_token_mapping = self.tokenizer.convert_tokens_to_ids(labels_text)
        else:
            self.label_token_mapping = None
    # def fix_stupid_metric_device_bs(self):
    #     for m in self.train_metrics:
    #         getattr(self, "train_" + m).to("cpu")
    #     for m in self.valid_metrics:
    #         getattr(self, "valid_" + m).to("cpu")
    #         # getattr(self, "valid_" + m).device = "cpu"



    def get_loss(self, outputs, target):
        return outputs.loss

    def forward(self, input_ids, attention_mask, decoder_attention_mask=None, labels=None, targets=None):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        return output

    def get_class_logits(self, outputs):
        return outputs.logits[:, 1, self.label_token_mapping].detach().cpu()

    def training_step(self, batch, batch_size):
        """ training step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        targets = batch['target_class']
        if self.num_classes == 2:
            targets = targets.float().divide(2).long()
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
            targets=targets.flatten()
        )

        label_logits = self.get_class_logits(outputs)
        prediction = np.argmax(label_logits, axis=1).flatten()

        self.log_metrics(self.train_metrics, F.softmax(label_logits, dim=-1), targets.cpu().flatten(), is_end=False, train=True)

        loss = self.get_loss(outputs, targets)

        self.train_loss(loss * input_ids.shape[0])
        self.log(
            "train_loss", self.train_loss.compute(), prog_bar=True, logger=True, on_epoch=False, on_step=True
        )
        return {'loss': loss, 'prediction': prediction, 'target': targets.cpu().flatten()}

    def training_epoch_end(self, training_step_outputs):
        self.log_metrics(self.train_metrics, is_end=True, train=True)
        prediction = np.hstack([output['prediction'] for output in training_step_outputs])
        target = np.hstack([output['target'] for output in training_step_outputs])
        # print(f'TRAIN \n{classification_report(target, prediction, zero_division=1)}\n')

    def configure_optimizers(self):
        """ configure optimizers """
        return AdamW(self.parameters(), lr=self.lr)

    def log_metrics(self, metrics, pred=None, target=None, is_end=True, train=False):
        prefix = "train_" if train else "valid_"
        for metric_name in metrics:
            if not is_end:
                getattr(self, prefix + metric_name)(pred, target)
            self.log(prefix + metric_name, getattr(self, prefix + metric_name), logger=True, prog_bar=True)
            # if is_end:
            #     metric_name.reset()

    def forward_v(self, input_ids, attention_mask):
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask, max_length=self.max_len,
            return_dict_in_generate=True, output_scores=True
        )
        return outputs

    def get_class_logits_v(self, outputs):
        logits = outputs.scores[0]
        label_logits = logits[:, self.label_token_mapping].detach().cpu()
        return label_logits

    def validation_step(self, batch, batch_size):
        """ validation step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]

        outputs = self.forward_v(input_ids=input_ids,
                       attention_mask=attention_mask)

        # encoder_outputs = self.model.encoder(input_ids, return_dict=True,
        #                                output_hidden_states=True)
        # decoder_input_ids = torch.tensor([[self.model._get_decoder_start_token_id()]], device=self.device)
        # generated = self.model.greedy_search(decoder_input_ids, encoder_outputs=encoder_outputs,
        #                                return_dict_in_generate=True, output_scores=True)

        label_logits = self.get_class_logits_v(outputs)
        prediction = np.argmax(label_logits, axis=1).flatten()
        targets = batch['target_class'].cpu().flatten()
        if self.num_classes == 2:
            targets = targets.float().divide(2).long()
        self.log_metrics(self.valid_metrics, F.softmax(label_logits, dim=-1), targets, is_end=False, train=False)
        # self.log(
        #     "val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        # )
        return {'prediction': prediction, "target": targets}

    def validation_epoch_end(self, validation_step_outputs):
        self.log_metrics(self.valid_metrics, is_end=True, train=False)
        prediction = np.hstack([output['prediction'] for output in validation_step_outputs])
        target = np.hstack([output['target'] for output in validation_step_outputs])
        print()
        print(f'\nVALID Epoch: [{self.current_epoch}]\n{classification_report(target, prediction, zero_division=1)}\n')
        # print(f'\nVALID Epoch: [{self.current_epoch}]\n{classification_report(target, prediction, zero_division=1)}\n', file=)

