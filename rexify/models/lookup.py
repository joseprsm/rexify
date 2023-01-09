from abc import abstractmethod

import numpy as np
import tensorflow as tf


class _BaseLookupModel(tf.keras.Model):
    def __init__(self, ids: np.ndarray, values: np.ndarray):
        super().__init__()
        self._ids = ids
        self._values = values

        identifiers_idx = np.arange(0, self._ids.shape[0])
        init = tf.lookup.KeyValueTensorInitializer(
            keys=self._ids,
            values=identifiers_idx,
            key_dtype=tf.int32,
            value_dtype=tf.int32,
        )

        self.token_to_id = tf.lookup.StaticHashTable(init, default_value=len(ids))

    @tf.function(input_signature=[tf.TensorSpec([None], tf.int32)])
    def call(self, inputs):
        ids = self.token_to_id.lookup(inputs)
        return tf.nn.embedding_lookup(params=self._values, ids=ids)

    @abstractmethod
    def get_config(self):
        pass


class EmbeddingLookup(_BaseLookupModel):
    def __init__(self, ids: np.ndarray, embeddings: np.ndarray):
        super().__init__(ids=ids, values=embeddings)

    def get_config(self):
        return {"ids": self._ids, "embeddings": self._values}


class SessionLookup(_BaseLookupModel):
    def __init__(self, ids: np.ndarray, sessions: np.ndarray):
        super().__init__(ids=ids, values=sessions)

    def get_config(self):
        return {"ids": self._ids, "sessions": self._values}
