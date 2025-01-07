from typing import Optional

import pandas as pd

from relbench.base import BaseTask, Database, Table

from ctu_relational.datasets import CTUDataset


class CTUBaseTask(BaseTask):
    timedelta = pd.Timedelta(-1)

    dataset: CTUDataset

    # To be set by subclass is necessary.
    target_table: Optional[str] = None

    def get_sanitized_db(self, upto_test_timestamp: bool = True) -> Database:
        r"""Get the database object for the task without task target data.

        Args:
            upto_test_timestamp: If True, only return rows upto test_timestamp.
        Returns:
            Database: The database object.
        """

        raise NotImplementedError

    def make_table(self, db: Database, split: str) -> Table:
        r"""Make a table using the task definition.

        Args:
            db: The database object to use for (historical) ground truth.

        To be implemented by subclass. The table rows need not be ordered
        deterministically.
        """

        raise NotImplementedError

    def _get_table(self, split: str) -> Table:
        r"""Helper function to get a table for a split."""

        db = self.dataset.get_db(upto_test_timestamp=split != "test")

        table = self.make_table(db, split)
        table = self.filter_dangling_entities(table)

        return table
