import tensorflow as tf
import tensorflow_recommenders as tfrs

from rexify.models.base import DenseSetterMixin


class EventModel(tf.keras.Model, DenseSetterMixin):
    def __init__(self, layer_sizes: list[int] = None, n_dims: int = 1):
        super().__init__()
        self._layer_sizes = layer_sizes or [64, 32]
        self._n_dims = n_dims

        self.hidden_layers = self._set_dense_layers(layer_sizes)
        self.output_layer = tf.keras.layers.Dense(self._n_dims, activation="softmax")
        self.task = tfrs.tasks.Ranking(loss=tf.keras.losses.CategoricalCrossentropy())

    def call(self, inputs, labels):
        x = self._call_layers(self.hidden_layers, inputs)
        x = self.output_layer(x)
        return self.task(labels=labels, predictions=x)

    def get_config(self):
        return {
            "layer_sizes": self._layer_sizes,
            "n_dims": self._n_dims,
        }
