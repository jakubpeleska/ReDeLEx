from typing import Dict
from collections import defaultdict

import torch

from torch_geometric.data import HeteroData

import lightning as L

from torchmetrics.aggregation import MaxMetric, MinMetric, MeanMetric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score


from relbench.tasks import get_task
from relbench.base import TaskType


from experiments.nn.rdl_model import RDLModel
from experiments.nn.losses import (
    TableContrastiveLoss,
    EdgeContrastiveLoss,
    ContextContrastiveLoss,
)
from experiments.utils import get_loss, get_tune_metric


def get_metrics(dataset_name: str, task_name: str):
    task = get_task(dataset_name, task_name)

    if task.task_type == TaskType.REGRESSION:
        return {"mae": MeanAbsoluteError(), "mse": MeanSquaredError(), "r2": R2Score()}

    elif task.task_type == TaskType.BINARY_CLASSIFICATION:
        return {
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "f1": BinaryF1Score(),
            "roc_auc": BinaryAUROC(),
        }
    else:
        raise ValueError(f"Task type {task.task_type} is unsupported")


class PretrainingModel(torch.nn.Module):
    def __init__(self, backbone: RDLModel, channels: int, temperature: float = 1.0):
        super().__init__()
        self.backbone = backbone

        self.table_cl = TableContrastiveLoss(
            channels=channels, node_types=backbone.node_types, temperature=temperature
        )
        self.edge_cl = EdgeContrastiveLoss(
            channels=channels,
            edge_types=backbone.edge_types,
            temperature=temperature,
        )
        self.context_cl = ContextContrastiveLoss(
            channels=channels,
            node_types=backbone.node_types,
            edge_types=backbone.edge_types,
            temperature=temperature,
        )

    def forward(self, batch: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = self.backbone(batch)
        cor_dict = self.backbone(batch, tf_attr="cor_tf")

        tloss = self.table_cl(x_dict, cor_dict)
        eloss = self.edge_cl(batch, x_dict)
        closs = self.context_cl(batch, x_dict)
        return x_dict, {
            "table_loss": tloss,
            "edge_loss": eloss,
            "context_loss": closs,
        }


class LightningPretraining(L.LightningModule):
    def __init__(
        self,
        model: PretrainingModel,
        optimizer: torch.optim.Optimizer,
        model_save_path: str,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.model_save_path = model_save_path
        self.train_loss_dict = defaultdict(MeanMetric)
        self.val_loss_dict = defaultdict(MeanMetric)
        self.val_best_loss = MinMetric()
        self.val_best_loss.update(float("inf"))

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        x_dict, losses = self.model(batch)
        loss = sum(losses.values())

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("Loss is NaN or Inf, check the model and data.")
        metrics_dict = {
            "train_loss": loss.detach(),
            **{f"train_{k}": v.detach() for k, v in losses.items()},
        }
        for k, v in metrics_dict.items():
            self.train_loss_dict[k].update(v)

        self.log_dict(metrics_dict, prog_bar=True, batch_size=1)

        return loss

    def on_train_epoch_end(self):
        train_loss_metrics: dict[str, float] = {}
        for k, v in self.train_loss_dict.items():
            train_loss_metrics[f"{k}_epoch"] = v.compute().item()
            v.reset()
        self.log_dict(train_loss_metrics, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x_dict, losses = self.model(batch)
        loss = sum(losses.values())

        metrics_dict = {
            "val_loss": loss.detach(),
            **{f"val_{k}": v.detach() for k, v in losses.items()},
        }
        for k, v in metrics_dict.items():
            self.val_loss_dict[k].update(v)

        self.log_dict(metrics_dict, prog_bar=True, batch_size=1)

        return loss

    def on_validation_epoch_end(self):
        val_loss_metrics: dict[str, float] = {}

        val_loss = self.val_loss_dict["val_loss"].compute()
        best_val_loss = self.val_best_loss.compute()
        self.val_best_loss.update(val_loss)

        for k, v in self.val_loss_dict.items():
            val_loss_metrics[f"{k}_epoch"] = v.compute()
            v.reset()
            if val_loss < best_val_loss:
                val_loss_metrics[f"best_{k}"] = val_loss_metrics[f"{k}_epoch"]

        if val_loss < best_val_loss:
            torch.save(self.model.backbone.state_dict(), self.model_save_path)

        self.log_dict(val_loss_metrics, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return self.optimizer


class LightningEntityTaskModel(L.LightningModule):
    def __init__(
        self,
        backbone: RDLModel,
        head: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataset_name: str,
        task_name: str,
        finetune_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        if finetune_backbone:
            self.backbone.train()
        else:
            self.backbone.eval()
            self.backbone.requires_grad_(False)

        self.head = head

        self.task = get_task(dataset_name, task_name)
        self.loss_fn, _ = get_loss(dataset_name, task_name)
        self.metrics = get_metrics(dataset_name, task_name)
        self.tune_metric, self.higher_is_better = get_tune_metric(dataset_name, task_name)
        self.optimizer = optimizer

        self.train_loss = MeanMetric().requires_grad_(False)
        self.val_metric_dict = defaultdict(MeanMetric)
        self.best_tune_metric = MaxMetric() if self.higher_is_better else MinMetric()
        self.best_tune_metric.requires_grad_(False)
        if self.higher_is_better:
            self.best_tune_metric.update(float("-inf"))
        else:
            self.best_tune_metric.update(float("inf"))

    def forward(self, batch):
        x_dict = self.backbone(
            batch,
            self.task.entity_table,
        )
        pred = self.head(x_dict[self.task.entity_table])
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        if pred.size(0) != batch[self.task.entity_table].batch_size:
            pred = pred[: batch[self.task.entity_table].batch_size]

        if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            target = batch[self.task.entity_table].y.long()
        else:
            target = batch[self.task.entity_table].y.float()
        return pred, target

    def training_step(self, batch, batch_idx):
        pred, target = self(batch)
        loss = self.loss_fn(pred.float(), target)
        batch_size = pred.size(0)

        self.train_loss.update(loss.detach(), batch_size)

        self.log("train_loss", loss.detach().item(), prog_bar=True, batch_size=batch_size)

        return loss

    def on_train_epoch_end(self):
        train_loss = self.train_loss.compute()
        self.train_loss.reset()
        self.log_dict({"train_loss_epoch": train_loss}, prog_bar=True, logger=True)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int, dataloader_idx: int):
        pred, target = self(batch)
        batch_size = pred.size(0)

        mode = "val" if dataloader_idx == 0 else "test"

        metrics_dict = {
            f"{mode}_{mname}": mfn(pred, target) for mname, mfn in self.metrics.items()
        }
        for k, v in metrics_dict.items():
            self.val_metric_dict[k].update(v, batch_size)

        self.log_dict(
            metrics_dict, prog_bar=True, batch_size=batch_size, add_dataloader_idx=False
        )

    def on_validation_epoch_end(self):
        val_metrics: dict[str, float] = {}

        tune_metric = self.val_metric_dict[f"val_{self.tune_metric}"].compute()
        best_tune_metric = self.best_tune_metric.compute()
        self.best_tune_metric.update(tune_metric)

        for k, v in self.val_metric_dict.items():
            val_metrics[f"{k}_epoch"] = v.compute()
            v.reset()
            if (self.higher_is_better and tune_metric > best_tune_metric) or (
                not self.higher_is_better and tune_metric < best_tune_metric
            ):
                val_metrics[f"best_{k}"] = val_metrics[f"{k}_epoch"]

        self.log_dict(val_metrics, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return self.optimizer
