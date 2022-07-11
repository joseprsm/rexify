import numpy as np

from time import mktime
from pandas import to_datetime
from sklearn.base import TransformerMixin, BaseEstimator


class TimestampTransformer(BaseEstimator, TransformerMixin):

    range: range

    def fit(self, X, *_):
        self.range = range(X.shape[1])
        return self

    def transform(self, X):
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
