from abc import abstractmethod
from pathlib import Path

import pandas as pd

from rexify.features.base import HasSchemaMixin
from rexify.schema import Schema


class BaseDataFrame(pd.DataFrame, HasSchemaMixin):
    def __init__(self, data: pd.DataFrame, schema: Schema) -> None:
        pd.DataFrame.__init__(self, data)
        HasSchemaMixin.__init__(self, schema=schema)

    @abstractmethod
    def load(cls, path: str | Path, **kwargs):
        pass
