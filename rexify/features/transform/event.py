from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

from rexify.features.io import HasSchemaInput
from rexify.types import Schema


class EventEncoder(BaseEstimator, TransformerMixin, HasSchemaInput):
    def __init__(self, schema: Schema):
        HasSchemaInput.__init__(self, schema)
        self._event_col = schema["event"]
        self._transformer = make_column_transformer(
            (OneHotEncoder(), [self._event_col])
        )

    @property
    def event_col(self):
        return self._event_col

    def fit(self, X, y=None, **fit_params):
        self._transformer.fit(X, y)
        return self

    def transform(self, X):
        oneh = self._transformer.transform(X)
        X = X.drop(self._event_col, axis=1)
        X[self._event_col] = oneh.tolist()
        return X
