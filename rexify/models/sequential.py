import tensorflow as tf

from rexify.models.base import DenseSetterMixin


class SequentialModel(tf.keras.Model, DenseSetterMixin):
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

        self.recurrent_model = self._set_recurrent_model()

        self.output_model = self._set_dense_layers(
            layer_sizes=self._dense_layer_sizes[:-1], activation=activation
        )
        self.output_model.append(tf.keras.layers.Dense(self._dense_layer_sizes[-1]))

    def call(self, inputs: tf.Tensor):
        x = tf.cast(inputs, tf.int32)
        x = self.embedding_layer(x)
        x = self._call_layers(self.recurrent_model, x)
        return self._call_layers(self.output_model, x)

    def _set_recurrent_model(self) -> tf.keras.Model:
        layer = getattr(tf.keras.layers, self._layer)
        layers = self._set_sequential_model(
            layer=layer,
            layer_sizes=self._recurrent_layer_sizes[:-1],
            return_sequences=True,
        )
        layers.append(layer(self._recurrent_layer_sizes[-1]))
        return layers

    def get_config(self):
        return {
            "n_dims": self._n_dims,
            "embedding_dim": self._embedding_dim,
            "layer": self._layer,
            "activation": self._activation,
            "recurrent_layer_sizes": self._recurrent_layer_sizes,
            "dense_layer_sizes": self._dense_layer_sizes,
        }
