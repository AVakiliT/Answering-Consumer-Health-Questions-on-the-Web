from typing import Optional, Any

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, STEP_OUTPUT
from torch_geometric import nn


class PipelineModule(pl.LightningModule):

    def __init__(self, model, tokenizer) -> None:
        super().__init__()
        self.qamodel = model
        self.tokenizer = tokenizer
        self.host_weight = nn.Linear(32, 1)


    def forward(self, *args, **kwargs) -> Any:
        return super().forward(*args, **kwargs)

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return super().training_step(*args, **kwargs)

    def training_step_end(self, *args, **kwargs) -> STEP_OUTPUT:
        return super().training_step_end(*args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return super().validation_step(*args, **kwargs)

    def validation_step_end(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return super().validation_step_end(*args, **kwargs)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
