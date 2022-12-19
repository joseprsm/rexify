from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from rexify import FeatureExtractor
from rexify.features.io import HasSchemaInput, SavableTransformer, TFDatasetGenerator
from rexify.features.transform import EventEncoder, Sequencer
from rexify.types import Schema


class EventGenerator(
    BaseEstimator, TransformerMixin, SavableTransformer, TFDatasetGenerator
):

    _user_id: list[str]
    _user_encoder: OrdinalEncoder

    _item_id: list[str]
    _item_encoder: OrdinalEncoder

    _model_params: dict[str, Any]

    def __init__(
        self,
        schema: Schema,
        timestamp: str,
        user_extractor: FeatureExtractor,
        item_extractor: FeatureExtractor,
        window_size: int = 3,
    ):
        self._timestamp = timestamp
        self._user_extractor = user_extractor
        self._item_extractor = item_extractor
        self._window_size = window_size

        self._model_params = self._get_model_params()

        self._ppl = make_pipeline(
            EventEncoder(schema),
            Sequencer(schema, timestamp_feature=timestamp, window_size=window_size),
        )

        HasSchemaInput.__init__(self, schema=schema)

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
        return self._ppl.transform(features)

    @staticmethod
    def encode(
        encoder: OrdinalEncoder, feature_names: list[str], data: pd.DataFrame
    ) -> pd.DataFrame:
        data[feature_names] = encoder.transform(data[feature_names])
        return data

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