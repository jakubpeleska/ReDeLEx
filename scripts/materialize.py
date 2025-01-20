from multiprocessing import Process
import os, sys
import traceback

sys.path.append("../")

import copy
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_frame import stype
from torch_frame.data import StatType
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset, get_dataset_names
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task, get_task_names

import ctu_relational
from ctu_relational.datasets import DBDataset
from ctu_relational.tasks import CTUEntityTask
from ctu_relational.utils import guess_schema, convert_timedelta, standardize_datetime

args = {
    "seed": 42,
    "cache_dir": "../.cache",
}

device = torch.device("cpu")
seed_everything(42)


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> torch.Tensor:
        return self.model.encode(sentences, convert_to_tensor=True)


def materialize_dataset(dataset_name: str):
    try:
        cache_path = Path(f"{args["cache_dir"]}/{dataset_name}")

        dataset: DBDataset = get_dataset(dataset_name)
        db = dataset.get_db(upto_test_timestamp=False)
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

            standardize_datetime(db, col_to_stype_dict)
            data, col_stats_dict = make_pkey_fkey_graph(
                db,
                col_to_stype_dict=col_to_stype_dict,
                text_embedder_cfg=TextEmbedderConfig(
                    text_embedder=GloveTextEmbedding(device=device), batch_size=256
                ),
                cache_dir=f"{cache_path}/materialized",
            )
    except Exception as e:
        with open(f"{cache_path}/error.txt", "w") as f:
            f.write(str(e))
            f.write(traceback.format_exc())
        print(f"Error: {e}")


ctu_datasets = list(filter(lambda x: x.startswith("ctu"), get_dataset_names()))

ps: List[Process] = []
for dataset_name in ctu_datasets:
    print(f"Processing {dataset_name}...")
    p = Process(target=materialize_dataset, args=(dataset_name,))
    ps.append(p)
    p.start()
    # p.join()

for p in ps:
    p.join()
