from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

from rexify.features.base import HasSchemaMixin
from rexify.schema import Schema


class EventEncoder(BaseEstimator, TransformerMixin, HasSchemaMixin):
    def __init__(self, schema: Schema):
        HasSchemaMixin.__init__(self, schema)
        self._event_type = schema.event_type
        self._transformer = make_column_transformer(
            (OneHotEncoder(), [self._event_type])
        )

    def fit(self, X, y=None, **fit_params):
        self._transformer.fit(X, y)
        return self

    def transform(self, X):
        oneh = self._transformer.transform(X)
        X = X.drop(self._event_type, axis=1)
        X[self._event_type] = oneh.tolist()
        return X
