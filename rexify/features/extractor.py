from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from rexify.features.io import HasSchemaInput, HasTargetInput
from rexify.features.transform import IDEncoder
from rexify.features.transformer import FeatureTransformer
from rexify.types import Schema


class _FeaturePipeline(tuple):
    def __new__(cls, schema: Schema, target: str):
        name = f"{target}_featureExtractor"
        ppl = make_pipeline(
            IDEncoder(schema, target),
            FeatureTransformer(schema, target),
        )
        keys = list(schema[target].keys())
        return tuple.__new__(_FeaturePipeline, (name, ppl, keys))


class FeatureExtractor(ColumnTransformer, HasSchemaInput, HasTargetInput):
    def __init__(self, schema: Schema, target: str):
        HasSchemaInput.__init__(self, schema)
        HasTargetInput.__init__(self, target)
        ColumnTransformer.__init__(self, [_FeaturePipeline(self._schema, self._target)])

    def apply(self, X):
        features = self.fit_transform(X)
        return features[:, :-1], features[:, -1]
