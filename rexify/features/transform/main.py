from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from rexify.features.base import HasSchemaInput
from rexify.features.transform.pipelines import CategoricalPipeline, NumericalPipeline
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

        categorical_ppl = CategoricalPipeline(self.schema, target)
        transformer_list += [categorical_ppl] if categorical_ppl != tuple() else []

        numerical_ppl = NumericalPipeline(self.schema, target)
        transformer_list += [numerical_ppl] if numerical_ppl != tuple() else []

        return transformer_list

    @property
    def use_sequential(self):
        return self._use_sequential
