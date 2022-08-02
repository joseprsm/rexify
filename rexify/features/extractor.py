from typing import Optional, Dict

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class FeatureExtractor(BaseEstimator, TransformerMixin):

    _ppl: Pipeline

    def __init__(
        self, schema: Dict[str, Dict[str, str]], transform: Optional[dict] = None
    ):
        self._schema = schema
        self._transform = transform

    def fit(self, X, y=None, **fit_params):
        self._ppl = self._make_pipeline()
        self._ppl.fit(X, y, **fit_params)
        return self

    def transform(self):
        raise NotImplementedError

    def _make_pipeline(self) -> Pipeline:
        raise NotImplementedError
