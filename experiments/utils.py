from typing import List, Optional

import json

from pathlib import Path

import torch
from torch.nn import BCEWithLogitsLoss, L1Loss, CrossEntropyLoss

from sentence_transformers import SentenceTransformer
from torch_frame import stype
from torch_frame.data import StatType
from torch_frame.config.text_embedder import TextEmbedderConfig

from torch_geometric.data import HeteroData

from relbench.base import EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.tasks import get_task
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    macro_f1,
    mae,
    micro_f1,
    mse,
    r2,
    roc_auc,
)

from ctu_relational.datasets import DBDataset
from ctu_relational.tasks import CTUBaseEntityTask
from ctu_relational.utils import (
    guess_schema,
    convert_timedelta,
    standardize_db_dt,
    merge_tf,
)


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> torch.Tensor:
        return self.model.encode(sentences, convert_to_tensor=True)


def get_cache_path(dataset_name: str, task_name: str, cache_dir: str):
    task = get_task(dataset_name, task_name)
    if isinstance(task, CTUBaseEntityTask):
        return Path(f"{cache_dir}/{dataset_name}/{task_name}")

    elif isinstance(task, EntityTask):
        return Path(f"{cache_dir}/{dataset_name}")

    else:
        raise ValueError(f"Task type {type(task)} is unsupported")


def get_metrics(dataset_name: str, task_name: str):
    task = get_task(dataset_name, task_name)

    if task.task_type == TaskType.REGRESSION:
        return [mae, mse, r2]

    elif task.task_type == TaskType.BINARY_CLASSIFICATION:
        return [accuracy, average_precision, f1, roc_auc]

    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return [accuracy, macro_f1, micro_f1]
    else:
        raise ValueError(f"Task type {task.task_type} is unsupported")


def get_tune_metric(dataset_name: str, task_name: str):
    task = get_task(dataset_name, task_name)

    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return "roc_auc", True

    elif task.task_type == TaskType.REGRESSION:
        return "mae", False

    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return "macro_f1", True
    else:
        raise ValueError(f"Task type {task.task_type} is unsupported")


def get_loss(dataset_name: str, task_name: str):
    task = get_task(dataset_name, task_name)

    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return BCEWithLogitsLoss(), 1

    elif task.task_type == TaskType.REGRESSION:
        return L1Loss(), 1

    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return CrossEntropyLoss(), len(task.stats[StatType.COUNT][0])

    else:
        raise ValueError(f"Task type {task.task_type} is unsupported")


def get_data(
    dataset_name: str,
    task_name: str,
    cache_path: str,
    entity_table_only: bool = False,
    aggregate_neighbors: bool = False,
):
    dataset = get_dataset(dataset_name)
    task = get_task(dataset_name, task_name)
    if isinstance(task, CTUBaseEntityTask):
        db = task.get_sanitized_db(upto_test_timestamp=False)
    else:
        db = dataset.get_db(upto_test_timestamp=False)

    convert_timedelta(db)

    stypes_cache_path = Path(f"{cache_path}/stypes.json")
    try:
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for tname, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                if isinstance(stype_str, str):
                    col_to_stype[col] = stype(stype_str)
    except FileNotFoundError:
        if isinstance(dataset, DBDataset):
            col_to_stype_dict = guess_schema(db, dataset.get_schema())
        else:
            col_to_stype_dict = guess_schema(db)
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

    if entity_table_only and aggregate_neighbors:
        edge_dict = data.collect("edge_index")
        node_tf = data[task.entity_table].tf
        for (src, edge_name, dst), edge_index in edge_dict.items():
            if (
                src == task.entity_table
                and edge_index[0].unique(return_counts=True)[1].max() == 1
            ):
                prefix = f"{edge_name}_"
                node_tf = merge_tf(
                    left_tf=node_tf,
                    right_tf=data[dst].tf,
                    left_idx=edge_index[0],
                    right_idx=edge_index[1],
                    right_prefix=prefix,
                )
                col_stats_dict[task.entity_table].update(
                    {f"{prefix}{k}": v for k, v in col_stats_dict[dst].items()}
                )
        data[task.entity_table].tf = node_tf

    if entity_table_only:
        return (
            task,
            HeteroData({task.entity_table: data[task.entity_table]}),
            {task.entity_table: col_stats_dict[task.entity_table]},
        )

    return task, data, col_stats_dict
