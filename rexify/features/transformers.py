from typing import Tuple, List

from abc import abstractmethod

import numpy as np

from sklearn.pipeline import make_pipeline, make_union

from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)

from rexify.features.date import DateEncoder
from rexify.features.timestamp import TimestampTransformer


class TupleTransformer(tuple):
    def __new__(cls, features: List[str], **kwargs):
        transformer = cls.get_transformer(**kwargs)
        return tuple.__new__(TupleTransformer, (transformer, features))

    @classmethod
    @abstractmethod
    def get_transformer(cls, **kwargs):
        pass


# noinspection PyTypeChecker
class NumericalTransformer(TupleTransformer):
    @classmethod
    def get_transformer(
        cls, feature_range: Tuple[int] = None, n_bins: int = 5, encode: str = "ordinal"
    ):

        feature_range = feature_range or (-1, 1)

        return make_union(
            MinMaxScaler(feature_range=feature_range),
            KBinsDiscretizer(n_bins=n_bins, encode=encode),
        )


# noinspection PyTypeChecker
class DateTransformer(TupleTransformer):
    @classmethod
    def get_transformer(
        cls,
        feature_range: Tuple[int] = None,
        n_bins: int = 5,
        encode: str = "ordinal",
        **encoder_args
    ):

        feature_range = feature_range or (-1, 1)

        return make_union(
            DateEncoder(**encoder_args),
            make_pipeline(
                TimestampTransformer(),
                make_union(
                    MinMaxScaler(feature_range=feature_range),
                    KBinsDiscretizer(n_bins=n_bins, encode=encode),
                ),
            ),
        )


class CategoricalTransformer(TupleTransformer):
    @classmethod
    def get_transformer(cls, **kwargs):
        return make_pipeline(OneHotEncoder(dtype=np.int64, handle_unknown="ignore"))


# noinspection PyTypeChecker
class EmbeddingTransformer(TupleTransformer):
    @classmethod
    def get_transformer(cls, **kwargs):
        return make_union(
            OrdinalEncoder(
                dtype=np.int64,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
        )
