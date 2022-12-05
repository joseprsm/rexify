from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

from rexify.features.io import HasSchemaInput, HasTargetInput, SavableTransformer
from rexify.features.transform import CategoricalEncoder, IDEncoder, NumericalEncoder
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
