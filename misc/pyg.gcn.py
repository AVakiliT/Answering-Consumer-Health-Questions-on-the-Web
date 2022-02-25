import os

from pytorch_lightning.callbacks import ModelCheckpoint
from tldextract import tldextract
from torch import nn, optim
from torch_geometric import nn as geom_nn
from torch_geometric import data as geom_data
import pytorch_lightning as pl
import torch
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import train_test_split_edges
import torch_geometric


# %%

gnn_layer_by_name = {"GCN": geom_nn.GCNConv, "GAT": geom_nn.GATConv, "GraphConv": geom_nn.GraphConv}

# Small function for printing the test scores
def print_results(result_dict):
    if "train" in result_dict:
        print("Train accuracy: %4.2f%%" % (100.0 * result_dict["train"]))
    if "val" in result_dict:
        print("Val accuracy:   %4.2f%%" % (100.0 * result_dict["val"]))
    print("Test accuracy:  %4.2f%%" % (100.0 * result_dict["test"]))

class GNNModel(nn.Module):
    def __init__(
            self,
            c_in,
            c_hidden,
            c_out,
            num_layers=2,
            layer_name="GCN",
            dp_rate=0.1,
            **kwargs,
    ):
        """
        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of "hidden" graph layers
            layer_name: String of the graph layer to use
            dp_rate: Dropout rate to apply throughout the network
            kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels, out_channels=c_out, **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for layer in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x


class NodeLevelGNN(pl.LightningModule):
    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        if model_name == "MLP":
            # self.model = MLPModel(**model_kwargs)
            raise NotImplementedError
        else:
            self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)

        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, "Unknown forward mode: %s" % mode

        loss = self.loss_module(x[mask], data.y[mask])
        acc = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
        return loss, acc

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log("test_acc", acc)


CHECKPOINT_PATH = "checkpoints"
AVAIL_GPUS = [0]


def train_node_classifier(model_name, dataset, **model_kwargs):
    pl.seed_everything(42)
    node_data_loader = geom_data.DataLoader(dataset, batch_size=1)

    # Create a PyTorch Lightning trainer
    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
        gpus=AVAIL_GPUS,
        max_epochs=200,
        progress_bar_refresh_rate=0,
    )  # 0 because epoch size is 1
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "NodeLevel%s.ckpt" % model_name)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything()
        model = NodeLevelGNN(
            model_name=model_name, c_in=dataset.num_node_features, c_out=dataset.num_classes, **model_kwargs
        )
        trainer.fit(model, node_data_loader, node_data_loader)
        model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    test_result = trainer.test(model, test_dataloaders=node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc, "val": val_acc, "test": test_result[0]["test_acc"]}
    return model, result


# %%

v = pd.read_csv("./data/filtered_vertices.tsv", sep="\t", header=None, names="ccid rdomain nhosts domain".split())
e = pd.read_csv("./data/filtered_edges.tsv", sep="\t", header=None, names="from_ccid to_ccid".split())
v["id"] = list(range(v.shape[0]))

edge_list = pd.merge(pd.merge(e, v, left_on="from_ccid", right_on="ccid"), v, left_on="to_ccid", right_on="ccid")[
    ["id_x", "id_y"]]

urls = pd.read_csv("data/2019qrels.urls.csv", sep=" ")

urls["domain"] = urls.url.apply(lambda x: '.'.join(tldextract.extract(x)[-2:]))

qrels = pd.read_csv("./data/2019qrels.txt", sep=' ', header=None, index_col=False,
                    names=['topic_id', 'iteration', 'doc_id', 'relevance', 'stance', 'credibility'])

qrels = pd.merge(urls, qrels, left_on="id", right_on="doc_id", how="inner")[["domain", "credibility"]]

qrels_with_judged_cred = qrels[qrels.credibility >= 0]

domains_with_judged_cred = qrels_with_judged_cred.groupby("domain").agg(count=("credibility", "count"),
                                                                        credibility=("credibility", "mean"))

v = pd.merge(v, domains_with_judged_cred, on="domain", how="left")


# %%
import torch_geometric as pyg
data = Data(
    x=torch.eye(v.shape[0]),
    edge_index=torch.tensor(edge_list.to_numpy().T),
    y=torch.tensor(v.credibility.gt(0).to_numpy()).long(),
    num_classes=2,
)

data = pyg.transforms.RandomNodeSplit("random")(data)
from torch_geometric.loader import DataLoader

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = data

dataset = MyOwnDataset("misc/data")

#%%


node_gnn_model, node_gnn_result = train_node_classifier(
    model_name="GNN", layer_name="GCN", dataset=dataset, c_hidden=16, num_layers=3, dp_rate=0.1
)
print_results(node_gnn_result)
