from typing import Optional, Dict, List

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from rexify.utils import get_target_id


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

    def transform(self, X):
        return self._ppl.transform(X)

    def _make_pipeline(self) -> Pipeline:
        id_features: List[str] = [
            get_target_id(self._schema, target) for target in ["user", "item"]
        ]
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
            )
        )
