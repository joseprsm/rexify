from typing import Tuple, List, Dict, Any

from abc import abstractmethod

import numpy as np

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline, make_union

from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)

from rexify.features.cyclical import CyclicalTransformer
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
        date_attrs: List[str] = None,
        time_attrs: List[str] = None,
        get_time: bool = False,
        drop_columns: bool = True,
    ):

        feature_range = feature_range or (-1, 1)
        date_encoder = DateEncoder(date_attrs, time_attrs, get_time, drop_columns)

        cyclical_features = [attr for attr in date_encoder.attrs if attr != "year"]
        cyclical_index = np.argwhere([a in cyclical_features for a in date_encoder.attrs]).reshape(-1)

        date_bins = {"month": 12, "day": 30, "hour": 24, "minute": 60, "second": 60}

        return make_union(
            make_pipeline(
                date_encoder,
                make_column_transformer((
                    make_union(
                        *[
                            make_column_transformer((
                                CyclicalTransformer(num_bins=date_bins[feature], drop_columns=False), [i]
                            )) for i, feature in enumerate(cyclical_features)
                        ]
                    ),
                    cyclical_index),
                    remainder="passthrough",
                ),
            ),
            make_pipeline(
                TimestampTransformer(),
                make_union(
                    MinMaxScaler(feature_range=feature_range),
                    make_pipeline(
                        KBinsDiscretizer(n_bins=n_bins, encode=encode),
                        OneHotEncoder(dtype=np.int64)
                    )
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


class PreprocessingPipeline(ColumnTransformer):
    def __init__(
        self,
        embedding_args: Dict[str, Any] = None,
        numerical_args: Dict[str, Any] = None,
        date_args: Dict[str, Any] = None,
        categorical_args: Dict[str, Any] = None,
    ):
        self.embedding_args = embedding_args.copy()
        self.numerical_args = numerical_args.copy()
        self.date_args = date_args.copy()
        self.categorical_args = categorical_args.copy()

        self.embedding_features = embedding_args.pop('features')
        self.numerical_features = numerical_args.pop('features')
        self.date_features = date_args.pop('features')
        self.categorical_features = categorical_args.pop('features')

        super(PreprocessingPipeline, self).__init__(
            transformers=[
                ("embedding_pipeline",) + EmbeddingTransformer(self.embedding_features, **embedding_args),
                ("numerical_pipeline",) + NumericalTransformer(self.numerical_features, **numerical_args),
                ("date_pipeline",) + DateTransformer(self.date_features, **date_args),
                ("categorical_pipeline",)
                + CategoricalTransformer(self.categorical_features, **categorical_args),
            ]
        )

    def get_feature_names_out(self, input_features=None):
        return (
            self.embedding_features
            + self._set_feature_name("scaled", self.numerical_features)
            + self._set_feature_name("bucket", self.numerical_features)
            + self._get_date_feature_names()
            + self._get_categorical_feature_names()
        )

    @staticmethod
    def _set_feature_name(prefix: str, features: List[str]):
        return [f"{prefix}_{feature}" for feature in features]

    def _get_date_feature_names(self):

        base_features = self.transformers_[2][2]
        date_encoder = self.transformers_[2][1].transformer_list[0][1]

        def get_date_encoder_features():
            date_features = self.date_features if not date_encoder.drop_columns else []
            date_features += date_encoder.attrs
            return date_features

        def get_timestamp_features():
            features_ = [f"timestamp_{f}" for f in base_features]
            return self._set_feature_name("scaled", features_) + self._set_feature_name(
                "bucket", features_
            )

        return get_date_encoder_features() + get_timestamp_features()

    def _get_categorical_feature_names(self):
        return self.transformers_[3][1].steps[0][1].get_feature_names_out().tolist()
