from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

from rexify.features.io import HasSchemaInput, HasTargetInput, SavableTransformer
from rexify.features.transform import (
    CategoricalEncoder,
    EventEncoder,
    IDEncoder,
    NumericalEncoder,
    Sequencer,
)
from rexify.types import Schema
from rexify.utils import get_target_id


class _FeatureTransformer(ColumnTransformer, HasSchemaInput, HasTargetInput):
    def __init__(self, schema: Schema, target: str):
        HasSchemaInput.__init__(self, schema=schema)
        HasTargetInput.__init__(self, target=target)
        transformers = self._get_transformers()
        ColumnTransformer.__init__(
            self, transformers=transformers, remainder="passthrough"
        )

    def _get_transformers(self) -> list[tuple[str, Pipeline, list[str]]]:
        transformer_list = []

        cat_encoder = CategoricalEncoder(self._schema, self._target).as_tuple()
        transformer_list += [cat_encoder] if cat_encoder[-1] != tuple() else []

        num_encoder = NumericalEncoder(self._schema, self._target).as_tuple()
        transformer_list += [num_encoder] if num_encoder[-1] != tuple() else []

        return transformer_list


class _FeaturePipeline(tuple):
    def __new__(cls, schema: Schema, target: str):
        name = f"{target}_featureExtractor"
        ppl = make_pipeline(
            IDEncoder(schema, target),
            _FeatureTransformer(schema, target),
        )
        keys = list(schema[target].keys())
        return tuple.__new__(_FeaturePipeline, (name, ppl, keys))


class FeatureExtractor(
    ColumnTransformer, HasSchemaInput, HasTargetInput, SavableTransformer
):

    _model_params: dict[str, Any]

    def __init__(self, schema: Schema, target: str):
        HasSchemaInput.__init__(self, schema)
        HasTargetInput.__init__(self, target)
        ColumnTransformer.__init__(self, [_FeaturePipeline(self._schema, self._target)])

    def fit(self, X, y=None):
        super().fit(X, y)
        self._model_params = self._get_model_params(X)
        return self

    def transform(self, X) -> pd.DataFrame:
        features = super(FeatureExtractor, self).transform(X)
        return pd.DataFrame(features[:, :-1], index=features[:, -1])

    @property
    def model_params(self):
        return self._model_params

    def _get_model_params(self, X):
        id_col = get_target_id(self._schema, self._target)[0]
        input_dims = int(X[id_col].nunique() + 1)

        return {
            f"{self._target}_id": id_col,
            f"{self._target}_dims": input_dims,
        }


class Extractor(BaseEstimator, TransformerMixin, HasSchemaInput, SavableTransformer):
    def __init__(
        self,
        schema: Schema,
        timestamp: str,
        user_extractor: FeatureExtractor,
        item_extractor: FeatureExtractor,
        window_size: int = 3,
    ):
        self._timestamp = timestamp
        self._user_extractor = user_extractor
        self._item_extractor = item_extractor
        self._window_size = window_size

        self._ppl = make_pipeline(
            EventEncoder(schema),
            Sequencer(schema, timestamp_feature=timestamp, window_size=window_size),
        )

        HasSchemaInput.__init__(self, schema=schema)

    def fit(self, X: pd.DataFrame, y=None):
        x_ = self.encode(self._user_extractor, X)
        x_ = self.encode(self._item_extractor, x_)
        self._ppl.fit(x_, y)
        return self

    def transform(self, X):
        x_ = self.encode(self._user_extractor, X)
        x_ = self.encode(self._item_extractor, x_)
        x_ = self._ppl.transform(x_)
        return x_

    def encode(self, extractor: FeatureExtractor, data: pd.DataFrame) -> pd.DataFrame:
        encoder = self._get_id_encoder(extractor)
        data[encoder.target_feature] = encoder.transformer.transform(
            data[[encoder.target_feature]]
        )
        return data

    @staticmethod
    def _get_id_encoder(extractor: FeatureExtractor):
        return extractor.transformers_[0][1].steps[0][1]

    @property
    def item_extractor(self):
        return self._item_extractor

    @property
    def user_extractor(self):
        return self._user_extractor

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def window_size(self):
        return self._window_size
