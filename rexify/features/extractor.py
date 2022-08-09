from typing import Dict, Any, List

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from rexify.features.dataset import TfDatasetGenerator
from rexify.utils import get_target_ids, get_target_id


class FeatureExtractor(BaseEstimator, TransformerMixin, TfDatasetGenerator):

    _ppl: Pipeline
    _model_params: Dict[str, Any]
    _output_features: List[str]

    def __init__(self, schema: Dict[str, Dict[str, str]]):
        super(FeatureExtractor, self).__init__(schema=schema)

    def fit(self, X, y=None, **fit_params):
        self._ppl = self._make_pipeline()
        self._ppl.fit(X, y, **fit_params)
        self._model_params = self._get_model_params(X)
        self._output_features = X.columns
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

    def _get_model_params(self, X):
        user_id = get_target_id(self.schema, "user")
        user_input_dim = int(X[user_id].max() + 1)

        item_id = get_target_id(self.schema, "item")
        item_input_dim = int(X[item_id].max() + 1)

        return {
            "n_unique_items": item_input_dim,
            "item_id": item_id,
            "n_unique_users": user_input_dim,
            "user_id": user_id,
        }

    @property
    def model_params(self):
        return self._model_params

    @property
    def output_features(self):
        return self._output_features
