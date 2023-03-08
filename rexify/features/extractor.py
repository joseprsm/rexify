from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

from rexify.features.base import HasSchemaMixin, Serializable
from rexify.features.transform import CustomTransformer, EventEncoder, Sequencer
from rexify.features.transform.entity import EntityTransformer
from rexify.schema import Schema
from rexify.utils import get_target_id


class FeatureExtractor(BaseEstimator, TransformerMixin, HasSchemaMixin, Serializable):

    _model_params: dict[str, Any]

    def __init__(
        self,
        schema: Schema,
        users=None,
        items=None,
        return_dataset: bool = True,
        window_size: int = 3,
        custom_transformers: list[CustomTransformer] = None,
    ):
        HasSchemaMixin.__init__(self, schema)

        self._users = users
        self._items = items
        self._return_dataset = return_dataset
        self._window_size = window_size
        self._window_size = window_size
        self._timestamp = schema.timestamp
        self._custom_transformers = custom_transformers or []

        self._user_transformer = EntityTransformer(
            schema, "user", self._custom_transformers
        )
        self._item_transformer = EntityTransformer(
            schema, "item", self._custom_transformers
        )

        self._ppl = make_pipeline(
            EventEncoder(self._schema),
            Sequencer(
                self._schema,
                timestamp_feature=self._timestamp,
                window_size=self._window_size,
            ),
        )

    def fit(self, X: pd.DataFrame):
        _ = self._user_transformer.fit(self._users).transform(self._users)
        _ = self._item_transformer.fit(self._items).transform(self._items)

        x_ = X.copy()
        events = self._encode(self._user_transformer, x_)
        events = self._encode(self._item_transformer, events)
        _ = self._ppl.fit(events)

        self._model_params = self._get_model_params()
        return self

    def transform(self, X: pd.DataFrame):
        x_ = X.copy()
        events = self._encode(self._user_transformer, x_)
        events = self._encode(self._item_transformer, events)
        events = self._ppl.transform(events)
        events = self._drop(events, self._user_transformer)
        events = self._drop(events, self._item_transformer)
        self._model_params["session_history"] = self.history
        if self._return_dataset:
            return self.make_dataset(events)
        return events

    def make_dataset(self, X: pd.DataFrame) -> tf.data.Dataset:
        ds = self._get_dataset(X)
        ds = ds.map(self._get_header_fn())
        return ds

    @staticmethod
    def _encode(transformer: EntityTransformer, data: pd.DataFrame) -> pd.DataFrame:
        encoder, feature_names = transformer.encoder
        data[feature_names] = encoder.transform(data[feature_names])
        return data

    @staticmethod
    def _drop(df: pd.DataFrame, transformer: EntityTransformer):
        encoder, id_ = transformer.encoder
        return df.loc[df[id_].values.reshape(-1) != encoder.unknown_value, :]

    def _get_model_params(self):
        model_params = {}
        model_params.update(self._user_transformer.model_params)
        model_params.update(self._item_transformer.model_params)
        model_params.update({"ranking_features": self.ranking_features})
        model_params["window_size"] = self._window_size
        return model_params

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
                self.ranking_features[i]: x[i]
                for i in range(len(self.ranking_features))
            }

        return tf.data.Dataset.from_tensor_slices(
            data.loc[:, self.ranking_features].values.astype(np.int32)
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

    @property
    def users(self):
        return self._users

    @property
    def items(self):
        return self._items

    @property
    def load_fn(self):
        return self._load_fn

    @property
    def model_params(self):
        return self._model_params

    @property
    def ranking_features(self):
        return self._ppl.steps[0][1].ranking_features

    @property
    def history(self):
        return self._ppl.steps[1][1].history

    @property
    def return_dataset(self):
        return self._return_dataset

    @property
    def window_size(self):
        return self._window_size

    @property
    def custom_transformers(self):
        return self._custom_transformers
