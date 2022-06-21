import numpy as np
import torchmetrics
import pytorch_lightning as pl
from datasets import load_dataset
from sklearn.metrics import classification_report
from torch.optim import AdamW
import pandas as pd

YES = "▁yes"
NO = "▁no"
IRRELEVANT = "▁irrelevant"

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

def prep_boolq_dataset(prep_sentence, neg_sampling=True):
    dataset = load_dataset('super_glue', 'boolqstuff')
    df_train: pd.DataFrame
    df_validation: pd.DataFrame
    df_train, df_validation = [pd.concat({
        "source_text": dataset[sub].data.to_pandas().apply(
            lambda row: prep_sentence(row.question, row.passage), axis=1
        ),
        "target_text": dataset[sub].data.to_pandas().label.map({0: NO.replace("▁", ""), 1: YES.replace("▁", "")}),
        "target_class": dataset[sub].data.to_pandas().label.map({0: 0, 1: 2})
    }, axis=1) for sub in "train validation".split()]

    if neg_sampling:
        df_train_neg, df_validation_neg = [pd.concat({
            "source_text": pd.concat(
                [dataset[sub].data.to_pandas().question.shift(1), dataset[sub].data.to_pandas().passage],
                axis=1).iloc[1:].apply(
                lambda row: prep_sentence(row.question, row.passage), axis=1
            )
        }, axis=1) for sub in "train validation".split()]

        df_train_neg["target_text"] = IRRELEVANT.replace("▁", "")
        df_train_neg["target_class"] = 1
        df_validation_neg["target_text"] = IRRELEVANT.replace("▁", "")
        df_validation_neg["target_class"] = 1

        df_train = pd.concat([df_train, df_train_neg])
        df_validation = pd.concat([df_validation, df_validation_neg])

    return df_train, df_validation
