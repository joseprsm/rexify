import pandas as pd
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
        oneh = pd.DataFrame(oneh, columns=self.transformer.get_feature_names_out())
        x = X.drop(self._event_type, axis=1)
        return pd.concat([x, oneh], axis=1)

    @property
    def transformer(self):
        return self._transformer.transformers_[0][1]

    @property
    def ranking_features(self):
        return self.transformer.get_feature_names_out().tolist()
