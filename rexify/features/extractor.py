import pickle
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from rexify.features.dataset import TfDatasetGenerator
from rexify.features.pipelines import (
    CategoricalPipeline,
    IdentifierPipeline,
    NumericalPipeline,
    RankingPipeline,
)
from rexify.utils import get_target_id, make_dirs


class FeatureExtractor(BaseEstimator, TransformerMixin, TfDatasetGenerator):

    """Main transformer responsible for pre-processing event data.

    The input to this transformer should be the original dataset,
    in order to be pre-processed according to the pipeline. Additionally,
    the data schema should be passed in during instantiation. This
    transformer will infer a `sklearn.pipeline.Pipeline` and its steps
    to fit and transform the data, according to the passed schema.

    Args:
        schema (dict): a dictionary of dictionaries, corresponding to
            the user, item and context features

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

    _ppl: ColumnTransformer
    _model_params: dict[str, Any]
    _output_features: list[str]

    def __init__(self, schema: dict[str, dict[str, str]]):
        super(FeatureExtractor, self).__init__(schema=schema)

    def fit(self, X: pd.DataFrame, *_):
        """Fit FeatureExtractor to X.

        Args:
            X (pd.DataFrame): array-like of shape (n_samples, n_features)

        Returns:
            self: fitted encoder
        """
        self._ppl = self._make_transformer()
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

    def _make_transformer(self) -> ColumnTransformer:
        transformer_list: list[tuple[str, TransformerMixin, list[str]]] = [
            *self._get_features_transformers(target="user"),
            *self._get_features_transformers(target="item"),
        ]

        transformer_list += (
            [*self._get_features_transformers(target="context")]
            if "context" in self.schema.keys()
            else []
        )

        transformer_list += (
            self._get_features_transformers(target="rank")
            if "rank" in self.schema.keys()
            else []
        )

        return ColumnTransformer(transformer_list)

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
        }

    def _get_features_transformers(
        self, target: str
    ) -> List[Tuple[str, Pipeline, List[str]]]:
        transformer_list = []

        if target != "rank":
            if target != "context":
                transformer_list.append(IdentifierPipeline(self.schema, target))

            categorical_ppl = CategoricalPipeline(self.schema, target)
            transformer_list += [categorical_ppl] if categorical_ppl != tuple() else []

            numerical_ppl = NumericalPipeline(self.schema, target)
            transformer_list += [numerical_ppl] if numerical_ppl != tuple() else []
        else:
            transformer_list.append(RankingPipeline(self.schema, target))

        return transformer_list
