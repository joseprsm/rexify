from typing import List, Dict

import tensorflow as tf


class DateModel(tf.keras.Model):

    def __init__(self,
                 date_features: List[str],
                 embedding_dim: int = 32,
                 layer_sizes: List[int] = None,
                 activation: str = 'leaky_relu'):
        super(DateModel, self).__init__()
        self._activation = activation
        self._embedding_dim = embedding_dim
        self._date_features = date_features
        self._layer_sizes = layer_sizes if layer_sizes else [32, 16, 8]

        self.embedding_models = {
            key: tf.keras.Sequential([
                tf.keras.layers.Hashing(50),
                tf.keras.layers.Embedding(50, self._embedding_dim)
            ]) for key in self._date_features
        }

        self.dense_layers = [
            tf.keras.layers.Dense(num_units, activation=activation)
            for num_units in layer_sizes[:-1]
        ]

    def call(self, inputs: Dict[str, tf.Tensor], *_):
        x = tf.concat([
            self.embedding_models[k](v)
            for k, v in inputs.items() if k in self._date_features
        ], axis=1)

        for dense_layer in self.dense_layers:
            x = dense_layer(x)

        return x

    def get_config(self):
        return {
            'layer_sizes': self._layer_sizes,
            'date_features': self._date_features,
            'activation': self._activation,
            'embedding_dim': self._embedding_dim
        }
