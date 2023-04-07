from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

from rexify.features.base import HasSchemaMixin, HasTargetMixin
from rexify.features.transform import (
    CategoricalEncoder,
    CustomTransformer,
    IDEncoder,
    NumericalEncoder,
)
from rexify.schema import Schema
from rexify.utils import get_target_id


class _FeatureTransformer(ColumnTransformer, HasSchemaMixin, HasTargetMixin):
    def __init__(self, schema: Schema, target: str):
        HasSchemaMixin.__init__(self, schema=schema)
        HasTargetMixin.__init__(self, target=target)
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


class EntityTransformer(ColumnTransformer, HasSchemaMixin, HasTargetMixin):
    _features: pd.DataFrame
    _model_params: dict[str, Any]

    def __init__(
        self,
        schema: Schema,
        target: str,
        custom_transformers: list[CustomTransformer] = None,
    ):
        HasSchemaMixin.__init__(self, schema)
        HasTargetMixin.__init__(self, target)
        self._custom_transformers = (
            self._filter_custom_transformers(custom_transformers, self._target) or []
        )
        transformers = [
            self._get_feature_pipeline(self._schema, self._target)
        ] + self._custom_transformers
        ColumnTransformer.__init__(self, transformers)

    def fit(self, X, y=None):
        super().fit(X, y)
        n_dims = self._get_n_dims(X)
        self._model_params = n_dims
        return self

    def transform(self, X) -> pd.DataFrame:
        self._features = super().transform(X)
        self._features = pd.DataFrame(
            self._features[:, :-1], index=self._features[:, -1]
        )
        self._features = pd.concat(
            [
                self._features,
                pd.DataFrame(np.zeros(self._features.shape[1])).transpose(),
            ],
            ignore_index=True,
        )

        self._model_params.update({f"{self._target}_embeddings": self._features})
        return self._features

    def _get_n_dims(self, X):
        id_col = get_target_id(self._schema, self._target)[0]
        input_dims = int(X[id_col].nunique() + 1)
        return {f"{self._target}_dims": input_dims}

    @staticmethod
    def _filter_custom_transformers(
        custom_transformers: list[CustomTransformer], target: str
    ):
        def target_from_name(x):
            return x[0].split("_")[0] == target

        return list(filter(target_from_name, custom_transformers))

    @staticmethod
    def _get_feature_pipeline(schema, target) -> tuple[str, Pipeline, list[str]]:
        name = f"{target}_featureExtractor"
        ppl = make_pipeline(
            IDEncoder(schema, target),
            _FeatureTransformer(schema, target),
        )
        target_keys = getattr(schema, target).to_dict()
        keys = [target_keys.pop("id")] + list(target_keys.keys())
        return name, ppl, keys

    @property
    def model_params(self):
        return self._model_params

    @property
    def identifiers(self):
        return self._features.index.values.astype(int)

    @property
    def encoder(self):
        encoder = self.transformers_[0][1].steps[0][1].transformer.transformers_[0]
        return encoder[1], encoder[-1]

    @property
    def custom_transformers(self):
        return self._custom_transformers
