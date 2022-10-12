from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder


class EventEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, event_col: str, **kwargs):
        super().__init__()
        self._event_col = event_col
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
