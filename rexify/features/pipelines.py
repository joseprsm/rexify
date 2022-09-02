import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from rexify.features.base import PassthroughTransformer
from rexify.utils import get_target_feature


class _BaseFeaturePipeline:

    dtype: str
    ppl: Pipeline
    pipeline_name: str

    def __new__(cls, schema, target) -> tuple[str, Pipeline, list[str]]:
        name = "_".join([target, cls.pipeline_name])
        target_features = cls._get_features(schema, target, cls.dtype)
        return name, cls.ppl, target_features

    @staticmethod
    def _get_features(schema, target, dtype):
        return get_target_feature(schema, target, dtype)


class IdentifierPipeline(_BaseFeaturePipeline):

    dtype = "id"

    pipeline_name = "idPipeline"

    ppl = make_pipeline(
        OrdinalEncoder(
            dtype=np.int64,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    )


class CategoricalPipeline(_BaseFeaturePipeline):

    dtype = "categorical"

    pipeline_name = "categoricalPipeline"

    ppl = make_pipeline(OneHotEncoder(sparse=False))


class NumericalPipeline(_BaseFeaturePipeline):

    dtype = "numerical"

    pipeline_name = "numericalPipeline"

    ppl = make_pipeline(MinMaxScaler(feature_range=(-1, 1)))


class RankingPipeline(_BaseFeaturePipeline):

    dtype = "rank"

    pipeline_name = "rankingPipeline"

    ppl = make_pipeline(PassthroughTransformer())

    @staticmethod
    def _get_features(schema, target, dtype):
        return [feature["name"] for feature in schema[dtype]]
