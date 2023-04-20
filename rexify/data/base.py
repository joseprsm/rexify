from abc import abstractmethod
from pathlib import Path

import pandas as pd

from rexify.features.base import HasSchemaMixin


class BaseDataFrame(pd.DataFrame, HasSchemaMixin):
    def __init__(self, data: pd.DataFrame) -> None:
        pd.DataFrame.__init__(self, data)

    @abstractmethod
    @classmethod
    def load(cls, path: str | Path, **kwargs):
        pass

    @abstractmethod
    def save(self, path: str | Path):
        pass
