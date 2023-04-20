from pathlib import Path

from sklearn.model_selection import train_test_split

from rexify.data.base import BaseDataFrame


class Input(BaseDataFrame):
    @classmethod
    def load(cls, path: str | Path, load_fn: str = "read_csv"):
        return cls(getattr(cls, load_fn)(path))

    def save(self, path: str | Path):
        pass

    def split(self, **kwargs):
        train, val = train_test_split(self, **kwargs)
        return Input(train, self.schema), Input(val, self.schema)
