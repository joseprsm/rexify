import tensorflow as tf


class SequentialModel(tf.keras.Model):
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

    @staticmethod
    def _set_recurrent_layers(layer: str, layer_sizes: list[int]) -> tf.keras.Model:
        layer = getattr(tf.keras.layers, layer)
        model = tf.keras.Sequential()
        for num_neurons in layer_sizes[:-1]:
            model.add(layer(num_neurons, return_sequences=True))
        model.add(layer(layer_sizes[-1]))
        return model

    @staticmethod
    def _set_dense_layers(
        layer_sizes: list[int], activation: str = "leaky_relu"
    ) -> tf.keras.layers.Layer:
        model = tf.keras.Sequential()
        for num_neurons in layer_sizes[:-1]:
            model.add(tf.keras.layers.Dense(num_neurons, activation=activation))
        model.add(tf.keras.layers.Dense(layer_sizes[-1]))
        return model

    def get_config(self):
        return {
            "n_dims": self._n_dims,
            "embedding_dim": self._embedding_dim,
            "layer": self._layer,
            "activation": self._activation,
            "recurrent_layer_sizes": self._recurrent_layer_sizes,
            "dense_layer_sizes": self._dense_layer_sizes,
        }
