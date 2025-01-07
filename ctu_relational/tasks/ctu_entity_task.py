from typing import Callable, Dict, List, Optional
from numpy.typing import NDArray

import numpy as np

from relbench.base import Database, Table

from .ctu_base_task import CTUBaseTask


class CTUEntityTask(CTUBaseTask):
    entity_col = "__PK__"
    entity_table: str
    target_col: str

    def make_table(self, db: Database, split: str) -> Table:
        time_col = db.table_dict[self.entity_table].time_col

        if self.target_table is not None:
            table = self.dataset.load_target_table()
        else:
            table = Table(
                df=db.table_dict[self.entity_table].df,
                fkey_col_to_pkey_table={self.entity_col: self.entity_table},
                pkey_col=None,
                time_col=time_col,
            )

        # TODO: data spliting
        if time_col is not None:
            if split == "train":
                table.df = table.df[table.df[time_col] < self.dataset.val_timestamp]
            elif split == "val":
                table.df = table.df[
                    (table.df[time_col] >= self.dataset.val_timestamp)
                    & (table.df[time_col] < self.dataset.test_timestamp)
                ]
            else:
                table.df = table.df[table.df[time_col] >= self.dataset.test_timestamp]

            table.df = table.df[[self.entity_col, time_col, self.target_col]]
        else:
            random_state = np.random.RandomState(seed=42)
            train_df = table.df.sample(frac=0.8, random_state=random_state)
            if split == "train":
                table.df = train_df
            else:
                table.df = table.df.drop(train_df.index)
                val_df = table.df.sample(frac=0.5, random_state=random_state)
                if split == "val":
                    table.df = val_df
                else:
                    table.df = table.df.drop(val_df.index)

            table.df = table.df[[self.entity_col, self.target_col]]

        return table

    def get_sanitized_db(self, upto_test_timestamp: bool = True) -> Database:
        db = self.dataset.get_db(upto_test_timestamp=upto_test_timestamp)

        if self.target_table is not None:
            return db

        db.table_dict[self.entity_table].df.drop(columns=[self.target_col], inplace=True)

        return db

    def evaluate(
        self,
        pred: NDArray,
        target_table: Optional[Table] = None,
        metrics: Optional[List[Callable[[NDArray, NDArray], float]]] = None,
    ) -> Dict[str, float]:
        if metrics is None:
            metrics = self.metrics

        if target_table is None:
            target_table = self.get_table("test", mask_input_cols=False)

        target = target_table.df[self.target_col].to_numpy()
        if len(pred) != len(target):
            raise ValueError(
                f"The length of pred and target must be the same (got "
                f"{len(pred)} and {len(target)}, respectively)."
            )

        return {fn.__name__: fn(target, pred) for fn in metrics}

    def filter_dangling_entities(self, table: Table) -> Table:
        db = self.dataset.get_db()
        num_entities = len(db.table_dict[self.entity_table])
        filter_mask = table.df[self.entity_col] >= num_entities

        if filter_mask.any():
            table.df = table.df[~filter_mask]

        return table
