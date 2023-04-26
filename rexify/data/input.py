from abc import abstractmethod
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from rexify.data.base import BaseDataFrame
from rexify.schema import Schema


class Input(BaseDataFrame):
    def __init__(self, data: pd.DataFrame, schema: Schema) -> None:
        super().__init__(data, schema)

    @classmethod
    def load(cls, path: str | Path, load_fn: str = "read_csv", schema: Schema = None):
        return cls(data=getattr(pd, load_fn)(path), schema=schema)

    def split(self, **kwargs):
        train, val = train_test_split(self, **kwargs)
        return self.__class__(train, self.schema), self.__class__(val, self.schema)

    @abstractmethod
    def generate(cls, n: int = 100):
        raise NotImplementedError


class Events(Input):
    def __init__(self, data: pd.DataFrame, schema: Schema) -> None:
        super().__init__(data, schema)


class Users(Input):
    def __init__(self, data: pd.DataFrame, schema: Schema) -> None:
        super().__init__(data, schema)


class Items(Input):
    def __init__(self, data: pd.DataFrame, schema: Schema) -> None:
        super().__init__(data, schema)
