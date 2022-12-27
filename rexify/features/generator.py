from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from rexify import FeatureExtractor
from rexify.dataclasses import Schema
from rexify.features.base import HasSchemaMixin, Serializable, TFDatasetGenerator
from rexify.features.transform import EventEncoder, Sequencer


class EventGenerator(
    BaseEstimator, TransformerMixin, HasSchemaMixin, Serializable, TFDatasetGenerator
):

    _user_id: list[str]
    _user_encoder: OrdinalEncoder

    _item_id: list[str]
    _item_encoder: OrdinalEncoder

    _model_params: dict[str, Any]

    def __init__(
        self,
        schema: Schema,
        user_extractor: FeatureExtractor,
        item_extractor: FeatureExtractor,
        window_size: int = 3,
    ):
        self._timestamp = schema.timestamp
        self._user_extractor = user_extractor
        self._item_extractor = item_extractor
        self._window_size = window_size

        self._model_params = self._get_model_params()

        self._ppl = make_pipeline(
            EventEncoder(schema),
            Sequencer(
                schema, timestamp_feature=self._timestamp, window_size=window_size
            ),
        )

        HasSchemaMixin.__init__(self, schema=schema)

    def fit(self, X: pd.DataFrame, y=None):
        x_ = X.copy()
        self._user_encoder, self._user_id = self._get_id_name_encoder(
            self._user_extractor
        )
        features = self.encode(self._user_encoder, self._user_id, x_)

        self._item_encoder, self._item_id = self._get_id_name_encoder(
            self._item_extractor
        )
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

    @staticmethod
    def _get_id_name_encoder(extractor: FeatureExtractor):
        encoder = extractor.transformers_[0][1].steps[0][1].transformer.transformers_[0]
        return encoder[1], encoder[-1]

    def _get_model_params(self):
        model_params = {}
        model_params.update(self._user_extractor.model_params)
        model_params.update(self._item_extractor.model_params)
        return model_params

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
    def model_params(self):
        return self._model_params
