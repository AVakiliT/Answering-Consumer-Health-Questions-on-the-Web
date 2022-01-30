from typing import Any

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT


class s22(LightningModule):
    def forward(self, *args, **kwargs) -> Any:
        return super().forward(*args, **kwargs)

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return super().training_step(*args, **kwargs)

    def __init__(self):
        super().__init__()
        self.qe = None
        self.ce = None
        self.qa = None

