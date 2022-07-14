from typing import List

import tensorflow as tf


class QueryModel(tf.keras.Model):
    def __init__(
        self,
        n_users: int,
        user_id: str,
        embedding_dim: int = 32,
        layer_sizes: List[int] = None,
    ):

        super(QueryModel, self).__init__()
        self._n_users = n_users
        self._user_id = user_id
        self._embedding_dim = embedding_dim
        self._layer_sizes = layer_sizes

        self.embedding_layer = tf.keras.layers.Embedding(n_users, embedding_dim)

        self.dense_layers = [
            tf.keras.layers.Dense(num_neurons) for num_neurons in layer_sizes
        ]

    def call(self, inputs: tf.Tensor) -> List[tf.Tensor]:
        x = self.embedding_layer(inputs[self._user_id])
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self):
        return {
            "n_items": self._n_users,
            "user_id": self._user_id,
            "embedding_dim": self._embedding_dim,
            "layer_sizes": self._layer_sizes,
        }
