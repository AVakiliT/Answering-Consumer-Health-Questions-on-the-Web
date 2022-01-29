from typing import Optional

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class s22DataModule(LightningDataModule):
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def __init__(self, batch_size=None, data_dir="./"):
        super().__init__(batch_size)

    def prepare_data(self) -> None:
        super().prepare_data()
        # Dataset

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)



