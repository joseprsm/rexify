import numpy as np

from time import mktime
from pandas import to_datetime
from sklearn.base import TransformerMixin, BaseEstimator


class TimestampTransformer(BaseEstimator, TransformerMixin):

    range: range

    def fit(self, X, *_):
        self.range = range(1) if len(X.shape) < 2 else range(X.shape[1])
        return self

    # noinspection PyPep8Naming
    def transform(self, X):
        X = X.values if type(X) != np.ndarray else X
        X = X.reshape(-1, 1) if self.range.stop == 1 else X

        return np.concatenate(
            [
                np.array(list(map(self.to_timestamp, to_datetime(X[:, i])))).reshape(
                    -1, 1
                )
                for i in self.range
            ],
            axis=1,
        )

    @staticmethod
    def to_timestamp(d):
        return mktime(d.timetuple())
