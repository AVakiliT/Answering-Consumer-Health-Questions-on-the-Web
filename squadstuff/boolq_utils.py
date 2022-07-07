import json

import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from torch import nn
from transformers import Trainer, TrainerCallback


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    cp = classification_report(labels, preds)
    print(cp)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.class_weights = torch.tensor(kwargs["class_weights"]).float()
        del kwargs["class_weights"]
        super().__init__(*args, **kwargs)


    def compute_loss(self, model, inputs, return_outputs=False):
        labels = torch.tensor(inputs.get("labels")).to(self.model.device)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(self.model.device))
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels)
        return (loss, outputs) if return_outputs else loss





class EvaluationCallback(TrainerCallback):
    def __init__(self, output_dir) -> None:
        super().__init__()
        self.output_dir = output_dir

    metrics = []

    def on_evaluate(self, args, state, control, **kwargs):
        print(kwargs['metrics'])
        self.metrics.append(kwargs['metrics'])
        with open(f"{self.output_dir}/metrics.txt", 'w') as f:
            json.dump(self.metrics, f, indent=4)


