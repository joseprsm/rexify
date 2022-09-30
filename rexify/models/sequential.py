import tensorflow as tf

from rexify.models.base import DenseModelMixin


class SequentialModel(tf.keras.Model, DenseModelMixin):
    def __init__(
        self,
        n_dims: int,
        embedding_dim: int,
        layer: str = "LSTM",
        activation: str = "relu",
        recurrent_layer_sizes: list[int] = None,
        dense_layer_sizes: list[int] = None,
    ):
        super().__init__()
        self._layer = layer
        self._n_dims = n_dims
        self._embedding_dim = embedding_dim
        self._activation = activation
        self._recurrent_layer_sizes = recurrent_layer_sizes or [32] * 2
        self._dense_layer_sizes = dense_layer_sizes or [32, 16]

        self.embedding_layer = tf.keras.layers.Embedding(
            self._n_dims, self._embedding_dim
        )
        self.recurrent_layers = self._set_recurrent_layers(
            self._layer, self._recurrent_layer_sizes
        )
        self.dense_layers = self._set_dense_layers(
            self._dense_layer_sizes, activation=activation
        )

    def call(self, inputs: tf.Tensor):
        x = self.embedding_layer(inputs)
        x = self.recurrent_layers(x)
        return self.dense_layers(x)

    def _set_recurrent_layers(
        self, layer: str, layer_sizes: list[int]
    ) -> tf.keras.Model:
        layer = getattr(tf.keras.layers, layer)
        model = self._set_sequential_model(
            layer, layer_sizes[:-1], return_sequences=True
        )
        model.add(layer(layer_sizes[-1]))
        return model

    def _set_dense_layers(
        self, layer_sizes: list[int], activation: str = "leaky_relu"
    ) -> tf.keras.layers.Layer:
        model = super()._set_dense_layers(layer_sizes[:-1], activation)
        model.add(tf.keras.layers.Dense(layer_sizes[-1]))
        return model
