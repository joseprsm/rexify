from abc import abstractmethod

import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from rexify.utils import get_target_feature


class _BaseTransformerTuple(tuple):

    ppl: Pipeline
    name: str

    def __new__(cls, schema, target):
        target_features = cls._get_features(schema, target)
        return tuple.__new__(
            _BaseTransformerTuple, (cls.name, cls.ppl, target_features)
        )

    @staticmethod
    @abstractmethod
    def _get_features(schema, target):
        raise NotImplementedError


class IdentifierPipeline(_BaseTransformerTuple):

    name = "idPipeline"

    ppl = make_pipeline(
        OrdinalEncoder(
            dtype=np.int64,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    )

    @staticmethod
    def _get_features(schema, target):
        return get_target_feature(schema, target, "id")


class CategoricalPipeline(_BaseTransformerTuple):

    name = "categoricalPipeline"

    ppl = make_pipeline(OneHotEncoder(sparse=False))

    @staticmethod
    def _get_features(schema, target):
        return get_target_feature(schema, target, "categorical")


class NumericalPipeline(_BaseTransformerTuple):

    name = "numericalPipeline"

    ppl = make_pipeline(MinMaxScaler(feature_range=(-1, 1)))

    @staticmethod
    def _get_features(schema, target):
        return get_target_feature(schema, target, "numerical")
