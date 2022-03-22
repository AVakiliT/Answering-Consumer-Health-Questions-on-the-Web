# %%
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
from torch.nn import functional as F
from boolq.BaseModules import ClassifierLightningModel


class PipelineDataset(Dataset):
    """  PyTorch Dataset class  """

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: PreTrainedTokenizer,
            source_max_token_len: int = 512,

    ):
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len

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

        source_domain_id = torch.tensor(data_row.domain_id)

        return dict(
            source_text_input_ids=source_text_encoding["input_ids"],
            source_text_attention_mask=source_text_encoding["attention_mask"],
            source_domain_id=source_domain_id,
            target_class=data_row["efficacy"]
        )


class PipelineDataModule(pl.LightningDataModule):
    """ PyTorch Lightning data class """

    def __init__(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            tokenizer: PreTrainedTokenizer,
            batch_size: int = 1,
            source_max_token_len: int = 512,
            num_workers: int = 1,

    ):
        super().__init__()

        self.train_dataset = None
        self.val_dataset = None
        self.train_df = train_df
        self.val_df = val_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = PipelineDataset(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,

        )
        self.val_dataset = PipelineDataset(
            self.val_df,
            self.tokenizer,
            self.source_max_token_len,
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

class PipelineModule(ClassifierLightningModel):

    def __init__(self, model, tokenizer, emb, weight, train_metrics, val_metrics, num_classes=2, lr=1e-5) -> None:
        super().__init__(weights=weight, train_metrics=train_metrics, valid_metrics=val_metrics, num_classes=num_classes, lr=lr)
        self.qamodel = model
        self.tokenizer = tokenizer
        self.host_emb =emb
        self.host_weight = nn.Linear(emb.embedding_dim, 1)

    def forward(self, input_ids, attention_mask, source_domain_id, labels):
        b = input_ids.shape[0]
        k = input_ids.shape[1]
        l = input_ids.shape[2]
        input_ids = input_ids.view(b*k,l) #(BK)L
        attention_mask = attention_mask.view(b*k,l)
        output = self.qamodel(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        qa_outs = output.logits.softmax(-1).sub(.5)[:, 1].reshape(b,k)
        embd = self.host_emb(source_domain_id)
        t = self.host_weight(embd).squeeze(-1)
        # t = 1
        a = qa_outs*t
        a = a.sum(-1)
        return a

    def get_loss(self, outputs, target):
        return F.cross_entropy(outputs.logits, target.flatten(), weight=self.weights.to(target.device),
                               reduction="mean")

    def training_step(self, batch, batch_size):
        """ training step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        source_domain = batch["source_domain_id"]
        targets = batch['target_class']
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            source_domain_id=source_domain,
            labels=targets.flatten()
        )
        loss = F.binary_cross_entropy_with_logits(outputs, targets.float().flatten(), weight=self.weights.to(targets.device) if self.weights else None,
                        reduction="mean")
        prediction = outputs.detach().cpu().gt(0).int()

        self.log_metrics(self.train_metrics, torch.sigmoid(outputs.detach().cpu()), targets.cpu().flatten(), is_end=False,
                         train=True)


        self.train_loss(loss * input_ids.shape[0])
        self.log(
            "train_loss", self.train_loss, prog_bar=True, logger=True, on_epoch=False, on_step=True
        )
        return {'loss': loss, 'prediction': prediction, 'target': targets.cpu()}

    def validation_step(self, batch, batch_size):
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        source_domain = batch["source_domain_id"]
        targets = batch['target_class']
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            source_domain_id=source_domain,
            labels=targets.flatten()
        )
        loss = F.binary_cross_entropy_with_logits(outputs, targets.float().flatten(), weight=self.weights.to(targets.device) if self.weights else None,
                        reduction="mean")
        prediction = outputs.detach().cpu().gt(0).int()

        self.log_metrics(self.valid_metrics, torch.sigmoid(outputs.detach().cpu()), targets.cpu().flatten(), is_end=False,
                         train=False)


        self.train_loss(loss * input_ids.shape[0])
        self.log(
            "valid_loss", self.train_loss, prog_bar=True, logger=True, on_epoch=False, on_step=True
        )
        return {'loss': loss, 'prediction': prediction, 'target': targets.cpu()}

    def configure_optimizers(self):
        """ configure optimizers """
        return AdamW(self.parameters(), lr=self.lr)