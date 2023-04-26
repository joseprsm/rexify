from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

from rexify.data import Events, Items, Output, Users
from rexify.features.base import HasSchemaMixin, Serializable
from rexify.features.transform import CustomTransformer, EventEncoder, Sequencer
from rexify.features.transform.entity import EntityTransformer
from rexify.schema import Schema


class FeatureExtractor(BaseEstimator, TransformerMixin, HasSchemaMixin, Serializable):

    _model_params: dict[str, Any]
    _item_ids: np.ndarray
    _user_ids: np.ndarray

    def __init__(
        self,
        schema: Schema,
        users: str = None,
        items: str = None,
        return_dataset: bool = False,
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

    def fit(self, X: Events):
        self._fit_transformer(Users)
        self._fit_transformer(Items)

        x_ = X.copy()
        events = self._encode(self._user_transformer, x_)
        events = self._encode(self._item_transformer, events)
        _ = self._ppl.fit(events)

        self._model_params = self._get_model_params()
        return self

    def transform(self, X: Events) -> Output:
        x_ = X.copy()
        events = self._encode(self._user_transformer, x_)
        events = self._encode(self._item_transformer, events)
        events = self._ppl.transform(events)
        events = self._drop(events, self._user_transformer)
        events = self._drop(events, self._item_transformer)
        self._model_params["session_history"] = self.history

        transformed = Output(
            data=events, schema=self._schema, ranking_features=self.ranking_features
        )

        self._user_ids = self._get_ids(transformed, self._user_transformer)
        self._item_ids = self._get_ids(transformed, self._item_transformer)

        return transformed.to_dataset() if self._return_dataset else transformed

    def _fit_transformer(self, inputs: Users | Items):
        input_name = inputs.__name__.lower()
        input_path: str = getattr(self, f"_{input_name}")
        transformer = getattr(self, f"_{input_name[:-1]}_transformer")
        x = inputs.load(input_path, schema=self._schema)
        transformer.fit(x).transform(x)

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

    @staticmethod
    def _get_ids(df: pd.DataFrame, transformer: EntityTransformer):
        return df.loc[:, transformer.encoder[1][0]].values.astype(np.int32)

    @property
    def users(self):
        return self._users

    @property
    def items(self):
        return self._items

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

    @property
    def item_encoder(self):
        return self._item_transformer.encoder[0]

    @property
    def item_ids(self):
        return self._item_ids

    @property
    def user_encoder(self):
        return self._user_transformer.encoder[0]

    @property
    def user_ids(self):
        return self._user_ids
