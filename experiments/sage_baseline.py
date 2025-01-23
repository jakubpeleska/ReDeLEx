from argparse import ArgumentParser
from multiprocessing import Process
from typing import Any, Dict, List, Optional

import copy, json, math, os, random, sys

from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

import ray
from ray import tune, train as ray_train
from ray.train.torch import enable_reproducibility
from ray.tune.schedulers import ASHAScheduler
from ray.tune.logger.aim import AimLoggerCallback
from ray.tune.logger.mlflow import MLflowLoggerCallback


import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from torch.nn import BCEWithLogitsLoss, L1Loss, CrossEntropyLoss

from sentence_transformers import SentenceTransformer
from torch_frame import stype
from torch_frame.data import StatType
from torch_frame.config.text_embedder import TextEmbedderConfig

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything


from relbench.base import BaseTask, Database, Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset, get_dataset_names
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task, get_task_names


sys.path.append(".")

import ctu_relational
from ctu_relational.datasets import DBDataset
from ctu_relational.tasks import CTUBaseEntityTask, CTUEntityTaskTemporal
from ctu_relational.utils import (
    guess_schema,
    convert_timedelta,
    standardize_db_dt,
    standardize_table_dt,
)
from ctu_relational.utils import TIMESTAMP_MAX, TIMESTAMP_MIN

from experiments.nn.sagegnn import SAGEModel

# from experiments.nn.dbformer import DBFormerModel


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> torch.Tensor:
        return self.model.encode(sentences, convert_to_tensor=True)


def get_model(
    data: HeteroData,
    col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
    num_layers: int,
    channels: int,
    out_channels: int,
) -> torch.nn.Module:
    return SAGEModel(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=num_layers,
        channels=channels,
        out_channels=out_channels,
        aggr="sum",
        norm="batch_norm",
    )


def get_task_info(task: CTUBaseEntityTask) -> Dict:
    clamp_min, clamp_max = None, None
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        loss_fn = BCEWithLogitsLoss()
        tune_metric = "roc_auc"
        higher_is_better = True

    elif task.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fn = L1Loss()
        tune_metric = "mae"
        higher_is_better = False
        # Get the clamp value at inference time
        train_table = task.get_table("train")
        clamp_min, clamp_max = np.percentile(
            train_table.df[task.target_col].to_numpy(), [2, 98]
        )
    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        out_channels = len(task.stats[StatType.COUNT][0])
        loss_fn = CrossEntropyLoss()
        tune_metric = "macro_f1"
        higher_is_better = True
    else:
        raise ValueError(f"Task type {task.task_type} is unsupported")

    is_temporal = isinstance(task, CTUEntityTaskTemporal)

    return {
        "is_temporal": is_temporal,
        "clamp_min": clamp_min,
        "clamp_max": clamp_max,
        "out_channels": out_channels,
        "loss_fn": loss_fn,
        "tune_metric": tune_metric,
        "higher_is_better": higher_is_better,
    }


def run_experiment(
    config: tune.TuneConfig,
    task: CTUBaseEntityTask,
    data: HeteroData,
    col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
):
    context = ray_train.get_context()
    experiment_dir = context.get_trial_dir()

    random_seed: int = config["seed"]
    lr: float = config["lr"]
    min_epochs: int = config["min_epochs"]
    batch_size: int = config["batch_size"]
    channels: int = config["channels"]
    num_layers: int = config["num_layers"]
    num_neighbors: int = config["num_neighbors"]
    max_steps_per_epoch: int = config["max_steps_per_epoch"]
    min_total_steps: int = config["min_total_steps"]
    num_workers: int = 0

    task_info = get_task_info(task)
    loss_fn = task_info["loss_fn"]
    clamp_min = task_info["clamp_min"]
    clamp_max = task_info["clamp_max"]
    higher_is_better = task_info["higher_is_better"]
    tune_metric = task_info["tune_metric"]
    is_temporal = task_info["is_temporal"]

    # enable_reproducibility(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cpu")

    resources = context.get_trial_resources().required_resources
    print(f"Resources: {resources}")
    if "gpu" in resources and resources["gpu"] > 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_num_threads(1)

    loader_dict: Dict[str, NeighborLoader] = {}

    for split in ["train", "val", "test"]:
        table = task.get_table(split, mask_input_cols=False)
        standardize_table_dt(table)
        table_input = get_node_train_table_input(table=table, task=task)
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[int(num_neighbors / 2**i) for i in range(num_layers)],
            time_attr="time" if is_temporal else None,
            input_nodes=table_input.nodes,
            input_time=table_input.time if is_temporal else None,
            transform=table_input.transform,
            batch_size=batch_size,
            temporal_strategy="uniform",
            shuffle=split == "train",
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    model = get_model(
        data, col_stats_dict, num_layers, channels, task_info["out_channels"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)

    def train(split: str = "train") -> float:
        model.train()

        loader = loader_dict[split]

        loss_accum = count_accum = 0
        steps = 0
        total_steps = min(len(loader), max_steps_per_epoch)
        for batch in tqdm(loader, total=total_steps):
            batch = batch.to(device)

            optimizer.zero_grad()
            pred = model(
                batch,
                task.entity_table,
            )
            pred = pred.view(-1) if pred.size(1) == 1 else pred

            if pred.size(0) != batch[task.entity_table].batch_size:
                pred = pred[: batch[task.entity_table].batch_size]

            if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                target = batch[task.entity_table].y.long()
            else:
                target = batch[task.entity_table].y.float()

            loss = loss_fn(pred.float(), target)
            loss.backward()
            optimizer.step()

            loss_accum += loss.detach().item() * pred.size(0)
            count_accum += pred.size(0)

            steps += 1
            if steps > max_steps_per_epoch:
                break

        return loss_accum / count_accum

    @torch.no_grad()
    def test(split: str) -> np.ndarray:

        loader = loader_dict[split]

        model.eval()

        pred_list = []
        for batch in tqdm(loader):
            batch = batch.to(device)
            pred = model(
                batch,
                task.entity_table,
            )

            if task.task_type == TaskType.REGRESSION:
                pred = torch.clamp(pred, clamp_min, clamp_max)

            if task.task_type in [
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTILABEL_CLASSIFICATION,
            ]:
                pred = torch.sigmoid(pred)

            pred = pred.view(-1) if pred.size(1) == 1 else pred

            if pred.size(0) != batch[task.entity_table].batch_size:
                pred = pred[: batch[task.entity_table].batch_size]

            pred_list.append(pred.detach().cpu())
        return torch.cat(pred_list, dim=0).numpy()

    training_time = 0
    best_val_metric = -math.inf if higher_is_better else math.inf

    epoch_steps = min(len(loader_dict["train"]), max_steps_per_epoch)
    n_epochs = max(math.ceil(min_total_steps / epoch_steps), min_epochs)

    val_table = task.get_table("val")

    model_checkpoint = os.path.join(experiment_dir, "best_model.pth")

    for epoch in range(1, n_epochs + 1):
        start = timer()
        train_loss = train()
        end = timer()

        training_time += end - start

        val_pred = test("val")
        val_metrics = task.evaluate(val_pred, val_table, metrics=task.metrics)

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_time": training_time,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        if (higher_is_better and val_metrics[tune_metric] >= best_val_metric) or (
            not higher_is_better and val_metrics[tune_metric] <= best_val_metric
        ):
            best_val_metric = val_metrics[tune_metric]
            # torch.save(model.state_dict(), model_checkpoint)

            test_pred = test("test")
            test_metrics = task.evaluate(test_pred, metrics=task.metrics)
            metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

        ray_train.report(metrics)


def get_task_config(dataset_name: str, task_name: str, cache_path: str) -> Dict:
    dataset: DBDataset = get_dataset(dataset_name)
    task: CTUBaseEntityTask = get_task(dataset_name, task_name)
    db = task.get_sanitized_db(upto_test_timestamp=False)
    convert_timedelta(db)

    stypes_cache_path = Path(f"{cache_path}/stypes.json")
    try:
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                if isinstance(stype_str, str):
                    col_to_stype[col] = stype(stype_str)
    except FileNotFoundError:
        col_to_stype_dict = guess_schema(db, dataset.get_schema())
        Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_cache_path, "w") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)

    standardize_db_dt(db, col_to_stype_dict)

    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=torch.device("cpu")), batch_size=256
        ),
        cache_dir=f"{cache_path}/materialized",
    )

    return task, data, col_stats_dict


def run_ray_tuner(
    dataset_name: str,
    task_name: str,
    ray_address: Optional[str] = None,
    ray_storage_path: Optional[str] = None,
    ray_experiment_name: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
    aim_repo: Optional[str] = None,
    num_samples: Optional[int] = 1,
    random_seed: int = 42,
    cache_dir: str = ".cache",
):

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=False)

    config = {
        "dataset_name": dataset_name,
        "task_name": task_name,
        "seed": tune.randint(0, 1000),
        "lr": 0.005,
        "min_epochs": 2,
        "batch_size": 512,
        "channels": 128,
        "num_layers": 2,
        "num_neighbors": 64,
        "max_steps_per_epoch": 1000,
        "min_total_steps": 100,
    }
    # scheduler = ASHAScheduler(max_t=max_num_epochs, grace_period=1, reduction_factor=2)
    scheduler = None

    task, data, col_stats_dict = get_task_config(
        dataset_name, task_name, Path(f"{cache_dir}/{dataset_name}/{task_name}")
    )

    if ray_experiment_name is None:
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        ray_experiment_name = f"sage_baseline_{time}_{dataset_name}_{task_name}"

    task_info = get_task_info(task)
    print(f"Task info: {task_info}")
    metric = f"val_{task_info["tune_metric"]}"
    metric_mode = "max" if task_info["higher_is_better"] else "min"

    train_table: pd.DataFrame = task.get_table("train").df
    use_gpu = len(train_table) > 40000

    ray_callbacks = []
    if mlflow_uri is not None:
        ray_callbacks.append(
            MLflowLoggerCallback(
                tracking_uri=mlflow_uri,
                experiment_name="pelesjak_sage_baseline",
            )
        )
    if aim_repo is not None:
        ray_callbacks.append(
            AimLoggerCallback(repo=aim_repo, experiment_name=ray_experiment_name)
        )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                run_experiment, task=task, data=data, col_stats_dict=col_stats_dict
            ),
            resources={"cpu": 1, "gpu": 0.25 if use_gpu else 0},
        ),
        run_config=ray_train.RunConfig(
            callbacks=ray_callbacks,
            name=ray_experiment_name,
            storage_path=ray_storage_path,
        ),
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=metric_mode,
            scheduler=scheduler,
            num_samples=num_samples,
            trial_name_creator=lambda trial: f"{ray_experiment_name}_{trial.trial_id}",
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result(metric, metric_mode)

    print("Best trial config: {}".format(best_result.config))
    print("Best trial test accuracy: {}".format(best_result.metrics["test_accuracy"]))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--ray_address", type=str, default="auto")
    parser.add_argument("--ray_storage", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--mlflow_uri", type=str, default=None)
    parser.add_argument("--aim_repo", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=1)

    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset
    task_name = args.task

    task: CTUBaseEntityTask = get_task(dataset_name, task_name)
    if task.task_type in [
        TaskType.LINK_PREDICTION,
        TaskType.MULTILABEL_CLASSIFICATION,
    ]:
        print(f"Skipping {dataset_name} - {task_name}...")

    else:
        print(f"Processing {dataset_name} - {task_name}...")

        run_ray_tuner(
            dataset_name,
            task_name,
            ray_address=args.ray_address,
            ray_storage_path=(
                os.path.realpath(args.ray_storage)
                if args.ray_storage is not None
                else os.path.realpath(".results")
            ),
            ray_experiment_name=args.run_name,
            mlflow_uri=args.mlflow_uri,
            aim_repo=args.aim_repo,
            num_samples=args.num_samples,
        )
