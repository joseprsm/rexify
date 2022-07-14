from typing import List

import tensorflow as tf


class CandidateModel(tf.keras.Model):
    def __init__(
        self,
        n_items: int,
        item_id: str,
        embedding_dim: int = 32,
        layer_sizes: List[int] = None,
    ):

        super(CandidateModel, self).__init__()
        self._item_id = item_id
        self._n_items = n_items
        self._embedding_dim = embedding_dim
        self._layer_sizes = layer_sizes

        self.embedding_layer = tf.keras.layers.Embedding(n_items, embedding_dim)

        self.dense_layers = [
            tf.keras.layers.Dense(num_neurons) for num_neurons in layer_sizes
        ]

    def call(self, inputs: tf.Tensor):
        x = self.embedding_layer(inputs[self._item_id])
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self):
        return {
            "n_items": self._n_items,
            "item_id": self._item_id,
            "embedding_dim": self._embedding_dim,
            "layer_sizes": self._layer_sizes,
        }
