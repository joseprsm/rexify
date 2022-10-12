from abc import ABC

import tensorflow as tf


class DenseSetterMixin(ABC):
    @staticmethod
    def _set_sequential_model(
        layer: str | tf.keras.layers.Layer, layer_sizes: list[int], **kwargs
    ) -> tf.keras.Sequential:
        if type(layer) == str:
            layer = getattr(tf.keras.layers, layer)
        model = tf.keras.Sequential()
        for num_neurons in layer_sizes:
            model.add(layer(num_neurons, **kwargs))
        return model

    def _set_dense_layers(
        self, layer_sizes: list[int], activation: str = "relu"
    ) -> tf.keras.layers.Layer:
        return self._set_sequential_model("Dense", layer_sizes, activation=activation)
