from abc import abstractmethod

import numpy as np
import tensorflow as tf

from rexify.models.base import DenseSetterMixin
from rexify.models.lookup import EmbeddingLookup


class TowerModel(tf.keras.Model, DenseSetterMixin):
    """

    Args:
        id_feature (str): the ID feature
        n_dims (str): number possible values for the ID feature
        embedding_dim (int): output dimension of the embedding layer
        layer_sizes (list): number of neurons in each layer for the output model
        feature_layers (list): number of neurons in each layer for the feature model

    Attributes:
         embedding_layer (tf.keras.layers.Embedding):
         feature_model (list):
         output_model (list):
    """

    def __init__(
        self,
        id_feature: str,
        n_dims: int,
        identifiers: np.array,
        feature_embeddings: np.array,
        embedding_dim: int = 32,
        layer_sizes: list[int] = None,
        feature_layers: list[int] = None,
    ):
        super().__init__()
        self._id_feature = id_feature
        self._n_dims = n_dims
        self._embedding_dim = embedding_dim
        self._layer_sizes = layer_sizes or [64, 32]
        self._feature_layers = feature_layers or [64, 32, 16]
        self._identifiers = identifiers
        self._target_features = feature_embeddings

        self.embedding_layer = tf.keras.layers.Embedding(n_dims, embedding_dim)
        self.feature_model = self._set_dense_layers(self._feature_layers)
        self.lookup_model = EmbeddingLookup(
            ids=self._identifiers, embeddings=self._target_features
        )
        self.output_model = self._set_dense_layers(self._layer_sizes, activation=None)

    @abstractmethod
    def call(self, inputs: dict[str, tf.Tensor], training: bool = None):
        raise NotImplementedError

    def get_config(self):
        return {
            "id_features": self._id_feature,
            "n_dims": self._n_dims,
            "embedding_dim": self._embedding_dim,
            "layer_sizes": self._layer_sizes,
            "feature_layers": self._feature_layers,
            "identifiers": self._identifiers,
            "feature_embeddings": self._target_features,
        }

    @property
    def identifiers(self):
        return self._identifiers
