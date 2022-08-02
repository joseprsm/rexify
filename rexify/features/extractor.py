from typing import Dict, Any

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from rexify.features.dataset import TfDatasetGenerator
from rexify.utils import get_target_ids


class FeatureExtractor(BaseEstimator, TransformerMixin, TfDatasetGenerator):

    _ppl: Pipeline
    _model_params: Dict[str, Any]

    def __init__(self, schema: Dict[str, Dict[str, str]]):
        super(FeatureExtractor, self).__init__(schema=schema)

    def fit(self, X, y=None, **fit_params):
        self._ppl = self._make_pipeline()
        self._ppl.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        return self._ppl.transform(X)

    def _make_pipeline(self) -> Pipeline:
        id_features = get_target_ids(self.schema)
        return make_pipeline(
            make_column_transformer(
                (
                    OrdinalEncoder(
                        dtype=np.int64,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    id_features,
                ),
                remainder="drop",
            ),
        )

    @property
    def model_params(self):
        return self._model_params
