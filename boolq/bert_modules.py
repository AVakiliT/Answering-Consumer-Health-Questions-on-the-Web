import torch
import torchmetrics
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import nn
from transformers import PreTrainedTokenizer, BertModel

from boolq.t5_modules import MyLightningModel
from torch.nn import functional as F


class BertLightningModel(MyLightningModel):
    """ PyTorch Lightning Model class"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.final = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, decoder_attention_mask=None, labels=None, targets=None):
        """ forward step """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=targets
        )
        return outputs

    # def training_step(self, batch, batch_size):
    #     """ training step """
    #     input_ids = batch["source_text_input_ids"]
    #     attention_mask = batch["source_text_attention_mask"]
    #     labels = batch["labels"]
    #     labels_attention_mask = batch["labels_attention_mask"]
    #
    #     outputs = self.forward(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         decoder_attention_mask=labels_attention_mask,
    #         labels=labels,
    #     )
    #
    #     logits = self.final(outputs[0][:, 0]).squeeze(-1)
    #
    #     prediction = logits.detach().cpu().sigmoid().multiply(2).floor().int()
    #     targets = batch['target_class'].flatten()
    #     loss = F.binary_cross_entropy_with_logits(logits, targets / 2, reduction='none')
    #     self.log_metrics(self.train_metrics, prediction / 2, targets.cpu().divide(2).int(), is_end=False)
    #     self.train_loss(loss.sum())
    #     # self.log('train_acc', self.train_acc.compute(), prog_bar=True, on_step=True, on_epoch=False)
    #
    #     self.log(
    #         "train_loss", self.train_loss.compute(), prog_bar=True, logger=True, on_epoch=False, on_step=True
    #     )
    #     return loss.mean()

    def get_class_logits(self, outputs):
        return outputs.logits.detach().cpu()

    def get_class_logits_v(self, outputs):
        return self.get_class_logits(outputs)

    def forward_v(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask)


    # def validation_step(self, batch, batch_size):
    #     """ validation step """
    #     input_ids = batch["source_text_input_ids"]
    #     attention_mask = batch["source_text_attention_mask"]
    #     labels = batch["labels"]
    #     labels_attention_mask = batch["labels_attention_mask"]
    #     targets = batch['target_class'].flatten()
    #     if self.num_classes == 2:
    #         targets = targets.float().divide(2).long()
    #     outputs = self.model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #     )
    #
    #     # encoder_outputs = self.model.encoder(input_ids, return_dict=True,
    #     #                                output_hidden_states=True)
    #     # decoder_input_ids = torch.tensor([[self.model._get_decoder_start_token_id()]], device=self.device)
    #     # generated = self.model.greedy_search(decoder_input_ids, encoder_outputs=encoder_outputs,
    #     #                                return_dict_in_generate=True, output_scores=True)
    #
    #     prediction = outputs.logits.detach().cpu().sigmoid().multiply(2).floor().int()
    #
    #     self.log_metrics(self.valid_metrics, F.softmax(outputs.logits, dim=-1), targets, is_end=False, train=False)
    #     # self.log(
    #     #     "val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
    #     # )
    #     return {'prediction': prediction, "target": targets}

    # def validation_epoch_end(self, validation_step_outputs):
    #     self.log('valid_epoch_accuracy', self.valid_acc.compute())
    #     self.log('valid_epoch_auroc', self.valid_auroc.compute())
    #     self.valid_acc.reset()
    #     self.valid_auroc.reset()
