from abc import abstractmethod

import tensorflow as tf


class TowerModel(tf.keras.Model):
    """

    Args:
        id_feature (str): the ID feature
        n_dims (str): number possible values for the ID feature
        embedding_dim (int): output dimension of the embedding layer
        layer_sizes (list): number of neurons in each layer for the output model
        feature_layers (list): number of neurons in each layer for the feature model

    Attributes:
         embedding_layer (tf.keras.layers.Embedding):
         feature_model (tf.keras.models.Sequential):
         output_model (tf.keras.models.Sequential):
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
        self.feature_model = self._get_dense_model(self._feature_layers)
        self.output_model = self._get_dense_model(self._layer_sizes)

    @abstractmethod
    def call(self, inputs: dict[str, tf.Tensor]):
        raise NotImplementedError

    @staticmethod
    def _get_dense_model(layer_sizes) -> tf.keras.Sequential:
        return tf.keras.Sequential(
            [tf.keras.layers.Dense(num_neurons) for num_neurons in layer_sizes]
        )

    def get_config(self):
        return {
            "id_features": self._id_feature,
            "n_dims": self._n_dims,
            "embedding_dim": self._embedding_dim,
            "layer_sizes": self._layer_sizes,
            "feature_layers": self._feature_layers,
        }
