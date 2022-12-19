import numpy as np
import tensorflow as tf


class EmbeddingLookup(tf.keras.Model):
    def __init__(
        self,
        identifiers: np.ndarray,
        embeddings: np.ndarray,
    ):
        super().__init__()
        self._identifiers = identifiers.reshape(-1)
        self._embeddings = embeddings

        identifiers_idx = np.arange(0, self._identifiers.shape[0])
        init = tf.lookup.KeyValueTensorInitializer(
            keys=self._identifiers,
            values=identifiers_idx,
            key_dtype=tf.int32,
            value_dtype=tf.int32,
        )

        self.token_to_id = tf.lookup.StaticHashTable(
            init, default_value=len(identifiers)
        )

    @tf.function(input_signature=[tf.TensorSpec([None], tf.int32)])
    def call(self, inputs):
        ids = self.token_to_id.lookup(inputs)
        return tf.nn.embedding_lookup(params=self._embeddings, ids=ids)

    def get_config(self):
        return {"identifiers": self._identifiers, "embeddings": self._embeddings}
