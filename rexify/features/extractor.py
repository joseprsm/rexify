from typing import Any, Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from rexify.features.base import HasSchemaMixin, Serializable
from rexify.features.transform import EventEncoder, Sequencer, TargetTransformer
from rexify.schema import Schema
from rexify.utils import get_target_id


class _EventGenerator(BaseEstimator, TransformerMixin, HasSchemaMixin):

    _user_id: list[str]
    _user_encoder: OrdinalEncoder

    _item_id: list[str]
    _item_encoder: OrdinalEncoder

    def __init__(
        self,
        schema: Schema,
        user_extractor: TargetTransformer,
        item_extractor: TargetTransformer,
        window_size: int = 3,
    ):
        self._timestamp = schema.timestamp
        self._user_extractor = user_extractor
        self._item_extractor = item_extractor
        self._window_size = window_size

        self._ppl = make_pipeline(
            EventEncoder(schema),
            Sequencer(
                schema, timestamp_feature=self._timestamp, window_size=window_size
            ),
        )

        HasSchemaMixin.__init__(self, schema=schema)

    def fit(self, X: pd.DataFrame, y=None):
        x_ = X.copy()
        self._user_encoder, self._user_id = self._user_extractor.encoder
        features = self.encode(self._user_encoder, self._user_id, x_)

        self._item_encoder, self._item_id = self._item_extractor.encoder
        features = self.encode(self._item_encoder, self._item_id, features)

        self._ppl.fit(features, y)
        return self

    def transform(self, X):
        x_ = X.copy()
        features = self.encode(self._user_encoder, self._user_id, x_)
        features = self.encode(self._item_encoder, self._item_id, features)
        features = self._ppl.transform(features)
        features = self.drop(features, "user")
        features = self.drop(features, "item")
        return features

    @staticmethod
    def encode(
        encoder: OrdinalEncoder, feature_names: list[str], data: pd.DataFrame
    ) -> pd.DataFrame:
        data[feature_names] = encoder.transform(data[feature_names])
        return data

    def drop(self, df: pd.DataFrame, target: str):
        id_ = getattr(self, f"_{target}_id")
        encoder = getattr(self, f"_{target}_encoder")
        return df.loc[df[id_].values.reshape(-1) != encoder.unknown_value, :]

    @property
    def item_extractor(self):
        return self._item_extractor

    @property
    def user_extractor(self):
        return self._user_extractor

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def window_size(self):
        return self._window_size

    @property
    def ranking_features(self):
        return self._ppl.steps[0][1].ranking_features


class _TFDatasetGenerator(HasSchemaMixin):
    _event_gen: _EventGenerator

    def make_dataset(self, X: pd.DataFrame) -> tf.data.Dataset:
        ds = self._get_dataset(X)
        ds = ds.map(self._get_header_fn())
        return ds

    def _get_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        return tf.data.Dataset.zip(
            (
                self._get_target_vector_dataset(data, self._schema, "user"),
                self._get_target_vector_dataset(data, self._schema, "item"),
                tf.data.Dataset.from_tensor_slices(
                    np.stack(data["history"].values).astype(np.int32)
                ),
                self._get_ranking_dataset(data),
            )
        )

    @staticmethod
    def _get_target_vector_dataset(
        data, schema: Schema, target: str
    ) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices(
            data.loc[:, get_target_id(schema, target)]
            .values.reshape(-1)
            .astype(np.int32)
        )

    def _get_ranking_dataset(self, data) -> tf.data.Dataset:
        @tf.autograph.experimental.do_not_convert
        def add_header(x):
            return {
                self._event_gen.ranking_features[i]: x[i]
                for i in range(len(self._event_gen.ranking_features))
            }

        return tf.data.Dataset.from_tensor_slices(
            data.loc[:, self._event_gen.ranking_features].values.astype(np.int32)
        ).map(add_header)

    @staticmethod
    def _get_header_fn():
        @tf.autograph.experimental.do_not_convert
        def header_fn(user_id, item_id, history, ranks):
            return {
                "query": {"user_id": user_id, "history": history},
                "candidate": {"item_id": item_id},
                "rank": ranks,
            }

        return header_fn


class FeatureExtractor(
    BaseEstimator, TransformerMixin, Serializable, _TFDatasetGenerator
):

    _event_gen: _EventGenerator
    _model_params: dict[str, Any]

    def __init__(
        self,
        schema: Schema,
        users_path: str = None,
        items_path: str = None,
        load_fn: Callable = pd.read_csv,
    ):
        HasSchemaMixin.__init__(self, schema)
        self._user_transformer = TargetTransformer(schema, "user")
        self._item_transformer = TargetTransformer(schema, "item")
        self._load_fn = load_fn

        self._users_path = users_path
        self._items_path = items_path

    def fit(self, X, y=None):
        self._fit_extractor("user")
        self._fit_extractor("item")
        self._event_gen = _EventGenerator(
            self._schema, self._user_transformer, self._item_transformer
        )
        self._event_gen.fit(X, y)
        self._model_params = self._get_model_params()
        return self

    def transform(self, X):
        return self._event_gen.transform(X)

    def _fit_extractor(self, target: str):
        path = getattr(self, f"_{target}s_path")
        transformer = getattr(self, f"_{target}_transformer")
        data = self._load_fn(path)
        _ = transformer.fit(data).transform(data)

    def _get_model_params(self):
        model_params = {}
        model_params.update(self._user_transformer.model_params)
        model_params.update(self._item_transformer.model_params)
        model_params.update({"ranking_features": self.ranking_features})
        return model_params

    @property
    def users_path(self):
        return self._users_path

    @property
    def items_path(self):
        return self._items_path

    @property
    def load_fn(self):
        return self._load_fn

    @property
    def model_params(self):
        return self._model_params

    @property
    def ranking_features(self):
        return self._event_gen.ranking_features
