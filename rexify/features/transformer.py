from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from rexify.features.io import HasSchemaInput
from rexify.features.transform import CategoricalEncoder, NumericalEncoder
from rexify.types import Schema


class FeatureTransformer(ColumnTransformer, HasSchemaInput):
    def __init__(self, schema: Schema, use_sequential: bool = True, **kwargs):
        self._use_sequential = use_sequential
        HasSchemaInput.__init__(self, schema=schema)
        transformers = self._get_transformers()
        ColumnTransformer.__init__(
            self, transformers=transformers, remainder="passthrough"
        )

    def _get_transformers(self) -> list:
        transformer_list: list[tuple[str, TransformerMixin, list[str]]] = [
            *self._get_features_transformers(target="user"),
            *self._get_features_transformers(target="item"),
        ]

        transformer_list += (
            [*self._get_features_transformers(target="context")]
            if "context" in self.schema.keys()
            else []
        )

        return transformer_list

    def _get_features_transformers(
        self, target: str
    ) -> list[tuple[str, Pipeline, list[str]]]:
        transformer_list = []

        cat_encoder = CategoricalEncoder(self.schema, target).as_tuple()
        transformer_list += [cat_encoder] if cat_encoder[-1] != tuple() else []

        num_encoder = NumericalEncoder(self.schema, target).as_tuple()
        transformer_list += [num_encoder] if num_encoder[-1] != tuple() else []

        return transformer_list

    @property
    def use_sequential(self):
        return self._use_sequential
