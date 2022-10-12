import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

from rexify.features.dataset import TFDatasetGenerator
from rexify.features.transform import FeatureTransformer, IDEncoder, Sequencer
from rexify.features.transform.event import EventEncoder
from rexify.types import Schema
from rexify.utils import (
    get_ranking_features,
    get_schema_features,
    get_target_id,
    make_dirs,
)


class FeatureExtractor(BaseEstimator, TransformerMixin, TFDatasetGenerator):

    """Main transformer responsible for pre-processing event data.

    The input to this transformer should be the original dataset,
    in order to be pre-processed according to the pipeline. Additionally,
    the data schema should be passed in during instantiation. This
    transformer will infer a `sklearn.pipeline.Pipeline` and its steps
    to fit and transform the data, according to the passed schema.

    Examples:

        >>> from rexify.features import FeatureExtractor

        >>> X = pd.DataFrame([['a', 1], ['b', 1]], columns=['user_id', 'item_id'])
        >>> schema_ = {"user": {"user_id": "id"}, "item": {"item_id": "id"}}

        >>> feat = FeatureExtractor(schema_)
        >>> feat.fit(X)
        FeatureExtractor(schema={'item': {'item_id': 'id'}, 'user': {'user_id': 'id'}})

        >>> X_ = feat.transform(X)
        >>> X_
        array([[0, 0],
               [1, 0]])

        >>> feat.make_dataset(X_)
        <MapDataset element_spec={'query': {'user_id': TensorSpec(shape=(), dtype=tf.int64, name=None), 'user_features': TensorSpec(shape=(0,), dtype=tf.float32, name=None), 'context_features': TensorSpec(shape=(0,), dtype=tf.float32, name=None)}, 'candidate': {'item_id': TensorSpec(shape=(), dtype=tf.int64, name=None), 'item_features': TensorSpec(shape=(0,), dtype=tf.float32, name=None)}}>
    """

    _model_params: dict[str, Any]
    _output_features: list[str]
    _rating_add: bool

    def __init__(self, schema: Schema, **kwargs):
        super().__init__(schema=schema)
        self._ppl = make_pipeline(
            IDEncoder(schema=schema),
            EventEncoder(**kwargs),
            Sequencer(schema=schema, **kwargs),
            FeatureTransformer(schema=schema, **kwargs),
        )

    def fit(self, X: pd.DataFrame, *_):
        """Fit FeatureExtractor to X.

        Args:
            X (pd.DataFrame): array-like of shape (n_samples, n_features)

        Returns:
            self: fitted encoder
        """
        self._validate_input(X)
        self._ppl.fit(X)
        self._model_params = self._get_model_params(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform X according to the inferred pipeline.

        Args:
            X (pd.DataFrame): The original event data

        Returns:
            numpy.ndarray: an array with the preprocessed features

        """
        return self._ppl.transform(X)

    def save(self, output_dir: str):
        make_dirs(output_dir)
        output_path = Path(output_dir) / "feat.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path):
        with open(path, "rb") as f:
            feat = pickle.load(f)
        return feat

    @property
    def model_params(self):
        return self._model_params

    def _get_model_params(self, X):
        user_id = get_target_id(self.schema, "user")[0]
        user_input_dim = int(X[user_id].nunique() + 1)

        item_id = get_target_id(self.schema, "item")[0]
        item_input_dim = int(X[item_id].nunique() + 1)

        return {
            "item_dims": item_input_dim,
            "item_id": item_id,
            "user_dims": user_input_dim,
            "user_id": user_id,
            "ranking_dims": X["event_type"].nunique(),
        }

    def _validate_input(
        self,
        X: pd.DataFrame,
    ):
        schema_columns = get_schema_features(self.schema)
        columns = ["event_type"] + schema_columns
        assert all([col in X.columns for col in columns])

        ranking_features = get_ranking_features(self.schema)
        assert all(
            [feat in X["event_type"].unique().tolist() for feat in ranking_features]
        )

    def _get_feature_names_out(self) -> list[str]:
        return self._ppl.steps[-1][-1].get_feature_names_out()
