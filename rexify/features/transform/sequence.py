import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from rexify.features.base import HasSchemaMixin
from rexify.schema import Schema
from rexify.utils import get_target_id


class Sequencer(BaseEstimator, TransformerMixin, HasSchemaMixin):

    """Transformer responsible for creating sequential data.

    It creates a new column `history` that holds the previous `window_size` event item IDs.

    Args:
        schema (rexify.types.Schema): the data schema
        timestamp_feature (str): the dataframe's feature name with a timestamp
        window_size (int): the size of the sliding window

    Examples:
        >>> from rexify.features.transform import Sequencer
        >>> sequencer = Sequencer(schema)
        >>> sequencer.fit(events)
        Sequencer(schema={'context': {'timestamp': 'timestamp'},
                      'item': {'item_id': 'id', 'price': 'numerical',
                               'type': 'categorical'},
                      'rank': [{'name': 'Purchase'}, {'name': 'Add to Cart'},
                               {'name': 'Page View'}],
                      'user': {'age': 'numerical', 'gender': 'categorical',
                               'user_id': 'id'}},
              timestamp_feature='timestamp', window_size=4)
        >>> transformed = sequencer.transform(events)

    """

    _user_id: str
    _item_id: str
    _columns: list[str]
    _padding: list[int]
    _history: pd.DataFrame

    def __init__(self, schema: Schema, window_size: int = 3, **kwargs):
        super().__init__(schema=schema)
        self._timestamp_feature = self._schema.timestamp
        self._window_size = window_size + 1

    def fit(self, X: pd.DataFrame, *_):
        self._user_id = get_target_id(self.schema, "user")[0]
        self._item_id = get_target_id(self.schema, "item")[0]
        self._columns = [col for col in X.columns if col != self._user_id]
        self._padding = [X[self._item_id].max() + 1] * (self._window_size - 2)
        return self

    def transform(self, X: pd.DataFrame):
        sequences = self._get_sequences(X)

        res = sequences.drop(self._item_id, axis=1).applymap(self._get_last)
        res[self._item_id] = sequences.pop(self._item_id)
        res["history"] = sequences.pop("history")
        res.reset_index(inplace=True)
        res = res.loc[res["history"].map(len) == self._window_size - 1, :]
        res = res.loc[~res.loc[:, self._timestamp_feature].isna()]

        self._history = self._get_history(res)

        res.drop(self._timestamp_feature, axis=1, inplace=True)
        return res

    def _get_sequences(self, df: pd.DataFrame):
        sequences: pd.DataFrame = (
            df.sort_values(self._timestamp_feature)
            .set_index(self._user_id)
            .groupby(level=-1)
            .apply(self._mask)
            .apply(pd.Series)
            .rename(columns=pd.Series(self._columns))
            .applymap(self._pad)
            .applymap(self._window)
            .apply(lambda x: x.explode())
        )

        sequences["history"] = sequences[self._item_id].map(lambda x: x[:-1])
        sequences[self._item_id] = sequences[self._item_id].map(self._get_last)
        return sequences

    def _get_history(self, df: pd.DataFrame):
        return (
            df.groupby([self._user_id])
            .agg({self._timestamp_feature: max, "history": list})
            .drop(self._timestamp_feature, axis=1)
            .history.map(self._get_last)
        )

    def _mask(self, df: pd.DataFrame):
        return [list(df[col]) for col in self._columns]

    @staticmethod
    def _get_last(lst: list):
        return lst[-1]

    def _window(self, sequence):
        if len(sequence) >= self._window_size:
            sequence = np.array(sequence, dtype=object)

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

    @property
    def history(self):
        return self._history
