import numpy as np
import torchmetrics
import pytorch_lightning as pl
from sklearn.metrics import classification_report
from torch.optim import AdamW


class ClassifierLightningModel(pl.LightningModule):

    def __init__(
            self,
            train_metrics,
            valid_metrics,
            num_classes,
            weights=None,
            save_only_last_epoch: bool = False,
            lr=1e-5
):

        super().__init__()
        self.lr = lr
        self.save_hyperparameters("num_classes", "lr")
        self.weights = weights
        for m in train_metrics:
            setattr(self, "train_" + m, getattr(torchmetrics, m)(num_classes=num_classes))
            # getattr(self, "train_" + m).device = "cpu"
        self.train_metrics = train_metrics
        for m in valid_metrics:
            setattr(self, "valid_" + m, getattr(torchmetrics, m)(num_classes=num_classes))
            # getattr(self, "valid_" + m).device = "cpu"
        self.valid_metrics = valid_metrics
        self.average_training_loss = None
        self.average_validation_loss = None
        self.save_only_last_epoch = save_only_last_epoch
        self.train_loss = torchmetrics.MeanMetric()
        self.valid_loss = torchmetrics.MeanMetric()

    def configure_optimizers(self):
        """ configure optimizers """
        return AdamW(self.parameters(), lr=self.lr)


    def training_epoch_end(self, training_step_outputs):
        self.log_metrics(self.train_metrics, is_end=True, train=True)
        prediction = np.hstack([output['prediction'] for output in training_step_outputs])
        target = np.hstack([output['target'] for output in training_step_outputs])

    def log_metrics(self, metrics, pred=None, target=None, is_end=True, train=False):
        prefix = "train_" if train else "valid_"
        for metric_name in metrics:
            if not is_end:
                getattr(self, prefix + metric_name)(pred, target)
            self.log(prefix + metric_name, getattr(self, prefix + metric_name), logger=True, prog_bar=True)
            # if is_end:
            #     metric_name.reset()



    def validation_epoch_end(self, validation_step_outputs):
        self.log_metrics(self.valid_metrics, is_end=True, train=False)
        prediction = np.hstack([output['prediction'] for output in validation_step_outputs])
        target = np.hstack([output['target'] for output in validation_step_outputs])
        print()
        print(f'\nVALID Epoch: [{self.current_epoch}]\n{classification_report(target, prediction, zero_division=1)}\n')
