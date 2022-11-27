import tensorflow as tf
import tensorflow_recommenders as tfrs

from rexify.models.base import DenseSetterMixin


class BaseRankingModel(tf.keras.Model, DenseSetterMixin):

    output_layer: tf.keras.layers.Dense
    task: tfrs.tasks.Ranking

    def __init__(self, layer_sizes: list[int]):
        super().__init__()
        self._layer_sizes = layer_sizes or [64, 32]
        self.hidden_layers = self._set_dense_layers(self._layer_sizes)

    def call(self, inputs, labels):
        x = self._call_layers(self.hidden_layers, inputs)
        x = self.output_layer(x)
        return self.task(labels=labels, predictions=x)
