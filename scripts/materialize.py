from typing import List, Optional

import json, os, sys, traceback

from multiprocessing import Process
from pathlib import Path

import torch

from sentence_transformers import SentenceTransformer

from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig

from relbench.base import BaseTask, TaskType
from relbench.datasets import get_dataset, get_dataset_names
from relbench.tasks import get_task_names, get_task
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal

sys.path.append(".")

import ctu_relational
from ctu_relational.datasets import DBDataset
from ctu_relational.tasks import CTUBaseEntityTask
from ctu_relational.utils import guess_schema, convert_timedelta, standardize_db_dt

args = {
    "seed": 42,
    "cache_dir": ".cache",
}


class GloveTextEmbedding:
    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=torch.device("cpu"),
        )

    def __call__(self, sentences: List[str]) -> torch.Tensor:
        return self.model.encode(sentences, convert_to_tensor=True)


def materialize_task_data(dataset_name: str, task_name: Optional[str] = None):
    try:
        dataset = get_dataset(dataset_name)

        if task_name is None:
            db = dataset.get_db(upto_test_timestamp=False)
            cache_path = Path(f"{args["cache_dir"]}/{dataset_name}")

        else:
            task: CTUBaseEntityTask = get_task(dataset_name, task_name)
            db = task.get_sanitized_db(upto_test_timestamp=False)
            cache_path = Path(f"{args["cache_dir"]}/{dataset_name}/{task_name}")

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
                    text_embedder=GloveTextEmbedding(), batch_size=256
                ),
                cache_dir=f"{cache_path}/materialized",
            )
    except Exception as e:
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        with open(f"{cache_path}/error.txt", "w") as f:
            f.write(str(e))
            f.write(traceback.format_exc())
        print(f"Error: {e}")


all_datasets = get_dataset_names()
ctu_datasets = list(filter(lambda x: x.startswith("ctu"), all_datasets))
relbench_datasets = list(filter(lambda x: x.startswith("rel"), all_datasets))

ps: List[Process] = []
for dataset_name in all_datasets:
    if dataset_name in relbench_datasets:
        print(f"Processing {dataset_name}...")
        p = Process(target=materialize_task_data, args=(dataset_name,))
        ps.append(p)
        p.start()
        continue

    for task_name in get_task_names(dataset_name):
        task = get_task(dataset_name, task_name)
        if task.task_type in [
            TaskType.LINK_PREDICTION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ]:
            print(f"Skipping {dataset_name} - {task_name}...")

        else:
            print(f"Processing {dataset_name}...")
            p = Process(target=materialize_task_data, args=(dataset_name, task_name))
            ps.append(p)
            p.start()
            # p.join()

for p in ps:
    p.join()
