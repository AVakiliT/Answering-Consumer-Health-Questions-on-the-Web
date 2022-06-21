import torch
import torchmetrics
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import nn
from transformers import PreTrainedTokenizer, BertModel

from boolqstuff.t5_modules import MyLightningModel
from torch.nn import functional as F


class BoolQBertModule(MyLightningModel):
    """ PyTorch Lightning Model class"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input_ids, attention_mask, decoder_attention_mask=None, labels=None, targets=None):
        """ forward step """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=targets
        )
        return outputs

    def get_loss(self, outputs, target):
        return F.cross_entropy(outputs.logits, target.flatten(), weight=self.weights.to(target.device), reduction="mean")

    def get_class_logits(self, outputs):
        return outputs.logits.detach().cpu()

    def get_class_logits_v(self, outputs):
        return self.get_class_logits(outputs)

    def forward_v(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask)

