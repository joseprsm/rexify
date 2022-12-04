from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from rexify.features.io import HasSchemaInput, HasTargetInput
from rexify.features.transform import CategoricalEncoder, NumericalEncoder
from rexify.types import Schema


class FeatureTransformer(ColumnTransformer, HasSchemaInput, HasTargetInput):
    def __init__(self, schema: Schema, target: str):
        HasSchemaInput.__init__(self, schema=schema)
        HasTargetInput.__init__(self, target=target)
        transformers = self._get_transformers()
        ColumnTransformer.__init__(
            self, transformers=transformers, remainder="passthrough"
        )

    def _get_transformers(self) -> list[tuple[str, Pipeline, list[str]]]:
        transformer_list = []

        cat_encoder = CategoricalEncoder(self.schema, self._target).as_tuple()
        transformer_list += [cat_encoder] if cat_encoder[-1] != tuple() else []

        num_encoder = NumericalEncoder(self.schema, self._target).as_tuple()
        transformer_list += [num_encoder] if num_encoder[-1] != tuple() else []

        return transformer_list
