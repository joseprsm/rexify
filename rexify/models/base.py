from abc import ABC

import tensorflow as tf


class DenseSetterMixin(ABC):
    @staticmethod
    def _set_sequential_model(
        layer: str | tf.keras.layers.Layer, layer_sizes: list[int], **kwargs
    ) -> list[tf.keras.layers.Layer]:
        if type(layer) == str:
            layer = getattr(tf.keras.layers, layer)
        return [layer(num_neurons, **kwargs) for num_neurons in layer_sizes]

    def _set_dense_layers(
        self, layer_sizes: list[int], activation: str | None = "relu"
    ) -> list[tf.keras.layers.Layer]:
        return self._set_sequential_model("Dense", layer_sizes, activation=activation)

    @staticmethod
    def _call_layers(layer_list: list[tf.keras.layers.Layer], inputs):
        x = inputs
        for layer in layer_list:
            x = layer(x)
        return x
