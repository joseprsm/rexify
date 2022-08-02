from typing import List

import numpy as np

from pandas import to_datetime

from sklearn.base import BaseEstimator, TransformerMixin


class DateEncoder(BaseEstimator, TransformerMixin):

    range: range

    def __init__(
        self,
        date_attrs: List[str] = None,
        time_attrs: List[str] = None,
        get_time: bool = False,
        drop_columns: bool = True,
    ):
        super(DateEncoder, self).__init__()
        self.get_time = get_time

        self.attrs = self.date_attrs = date_attrs or ["year", "month", "day"]

        self.time_attrs = time_attrs or ["hour", "minute", "second"]
        self.attrs += self.time_attrs if self.get_time else []

        self.drop_columns = drop_columns

    def fit(self, X, *_):
        self.range = range(1) if len(X.shape) < 2 else range(X.shape[1])
        return self

    # noinspection PyPep8Naming
    def transform(self, X):
        X = X.values if type(X) != np.ndarray else X
        X = X.reshape(-1, 1) if self.range.stop == 1 else X

        transformed_data = np.concatenate(
            [self._encode_dates(X[:, i]) for i in self.range], axis=1
        )

        transformed_data = (
            transformed_data
            if self.drop_columns
            else np.concatenate([X, transformed_data], axis=1)
        )

        return transformed_data

    def _encode_dates(self, features):
        return np.concatenate(
            [
                np.array(
                    list(map(self._get_date_attr(attr), to_datetime(features)))
                ).reshape(-1, 1)
                for attr in self.attrs
            ],
            axis=1,
        )

    @staticmethod
    def _get_date_attr(attr):
        def get_attribute(dt):
            return getattr(dt, attr)

        return get_attribute
