import tensorflow as tf


class CategoricalModel(tf.keras.Model):

    def __init__(self, input_dim: int, embedding_dim: int, **kwargs):
        super(CategoricalModel, self).__init__(**kwargs)
        self._input_dim = input_dim
        self._embedding_dim = embedding_dim
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=input_dim, output_dim=embedding_dim)

    def call(self, inputs, training=None, mask=None):
        return self.embedding_layer(inputs)

    def get_config(self):
        return {
            'input_dim': self._input_dim,
            'embedding_dim': self._embedding_dim
        }
