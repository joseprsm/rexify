from typing import List

import tensorflow as tf


class SequenceModel(tf.keras.Model):

    def __init__(self,
                 candidate_model,
                 layer_type: str = 'LSTM',
                 layer_sizes: List[int] = None):
        super(SequenceModel, self).__init__()
        self.candidate_model = candidate_model
        self._layer_type = layer_type
        self._layer_sizes = layer_sizes if layer_sizes is not None else [64, 32]
        self.recurrent_layers: List[tf.keras.Model] = self._get_recurrent_layers(
            layer_type, layer_sizes)

    def call(self, inputs, *_):
        x = self.candidate_model(inputs)
        for layer in self.recurrent_layers:
            x = layer(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'layer_type': self._layer_type,
            'layer_sizes': self._layer_sizes})
        return config

    @staticmethod
    def _get_recurrent_layers(layer_type: str,
                              layer_sizes: List[int]) -> List[tf.keras.Model]:

        def get_single_recurrent_layer(units, **kwargs):
            return getattr(tf.keras.layers, layer_type)(units=units, **kwargs)

        layers = list()
        if len(layer_sizes) > 1:
            layers = [
                get_single_recurrent_layer(num_units, return_sequences=True)
                for num_units in layer_sizes[:-1]]
        layers.append(get_single_recurrent_layer(units=layer_sizes[-1]))
        return layers
