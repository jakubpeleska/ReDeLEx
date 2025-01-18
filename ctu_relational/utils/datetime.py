from typing import Dict, Optional

import numpy as np
import pandas as pd

from torch_frame import stype

from relbench.base import Database


TIMESTAMP_MIN = np.datetime64(pd.Timestamp.min.date())
TIMESTAMP_MAX = np.datetime64(pd.Timestamp.max.date())


def convert_timedelta(db: Database):
    """Converts timedelta columns to datetime columns."""

    for table in db.table_dict.values():

        timedeltas = table.df.select_dtypes(include=["timedelta"])
        if not timedeltas.empty:
            timedeltas = pd.Timestamp("1900-01-01") + timedeltas
            table.df[timedeltas.columns] = timedeltas


def standardize_datetime(db: Database, col_to_stype_dict: Dict[str, Dict[str, stype]]):
    """Standartize datetime columns to UNIX timestamp (in datetime[ns] if possible)."""

    for tname, table in db.table_dict.items():
        for col in table.df.columns:
            if col_to_stype_dict[tname].get(col, None) == stype.timestamp:
                table.df[col] = table.df[col].astype(np.dtype("datetime64[ns]"))
