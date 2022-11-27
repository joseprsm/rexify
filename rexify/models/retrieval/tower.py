from abc import abstractmethod

import tensorflow as tf

from rexify.models.base import DenseSetterMixin


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

        self.embedding_layer = tf.keras.layers.Embedding(n_dims, embedding_dim)
        self.feature_model = self._set_dense_layers(self._feature_layers)
        self.output_model = self._set_dense_layers(self._layer_sizes, activation=None)

    @abstractmethod
    def call(self, inputs: dict[str, tf.Tensor]):
        raise NotImplementedError

    def get_config(self):
        return {
            "id_features": self._id_feature,
            "n_dims": self._n_dims,
            "embedding_dim": self._embedding_dim,
            "layer_sizes": self._layer_sizes,
            "feature_layers": self._feature_layers,
        }
