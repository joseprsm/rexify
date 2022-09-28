import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from rexify.features.base import HasSchemaInput
from rexify.types import Schema
from rexify.utils import get_target_id


class Sequencer(BaseEstimator, TransformerMixin, HasSchemaInput):

    _user_id: str
    _item_id: str
    _columns: list[str]
    _padding: list[int]

    def __init__(self, schema: Schema, timestamp_feature: str, window_size: int = 3):
        super().__init__(schema=schema)
        self._timestamp_feature = timestamp_feature
        self._window_size = window_size + 1

    def fit(self, X, *_):
        self._user_id = get_target_id(self.schema, "user")[0]
        self._item_id = get_target_id(self.schema, "item")[0]
        self._columns = [col for col in X.columns if col != self._user_id]
        self._padding = [-1] * (self._window_size - 2)
        return self

    def transform(self, X: pd.DataFrame):
        sequences = (
            X.sort_values(self._timestamp_feature)
            .set_index(self._user_id)
            .groupby(level=-1)
            .apply(self._mask)
            .apply(pd.Series)
        )

        sequences.columns = self._columns
        padded = pd.concat(
            [sequences[col].map(self._pad) for col in self._columns], axis=1
        )
        windowed = pd.concat(
            [padded[col].map(self._window) for col in self._columns], axis=1
        )
        exploded = pd.concat([windowed[col].explode() for col in self._columns], axis=1)
        exploded["history"] = exploded[self._item_id].map(lambda x: x[:-1])
        exploded[self._item_id] = exploded[self._item_id].map(self._get_last)

        res = pd.concat(
            [
                exploded[col].map(self._get_last)
                for col in self._columns
                if col != "item_id"
            ],
            axis=1,
        )
        res[self._item_id] = exploded[self._item_id]
        res["history"] = exploded["history"]

        return res

    def _mask(self, df: pd.DataFrame):
        return [list(df[col]) for col in self._columns]

    @staticmethod
    def _get_last(lst: list):
        return lst[-1]

    def _window(self, sequence):
        if len(sequence) >= self._window_size:
            sequence = np.array(sequence)

            stack = [
                sequence[range(i, i + self._window_size)]
                for i in range(len(sequence) - self._window_size + 1)
            ]

            if len(stack) > 1:
                stack = np.stack(stack)

            return stack
        return [sequence]

    def _pad(self, x: list):
        return self._padding + x

    @property
    def timestamp_feature(self):
        return self._timestamp_feature

    @property
    def window_size(self):
        return self._window_size
