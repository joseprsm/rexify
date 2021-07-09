from typing import Union

import numpy as np
import tensorflow as tf


class EmbeddingLookupModel(tf.keras.Model):

    def __init__(
            self,
            vocabulary: Union[tf.data.Dataset, np.array],
            embeddings: Union[tf.data.Dataset, np.array],
            **kwargs
    ):
        super(EmbeddingLookupModel, self).__init__(**kwargs)
        self.vocabulary = vocabulary
        self.embeddings = embeddings

        init = tf.lookup.KeyValueTensorInitializer(
            keys=self.vocabulary, values=list(range(len(vocabulary))))
        self.token_to_id = tf.lookup.StaticHashTable(init, default_value=len(vocabulary))

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def call(self, inputs, *_):
        tokens = tf.strings.split(inputs, sep=None)
        ids = self.token_to_id.lookup(tokens)
        embeddings = tf.nn.embedding_lookup(
            params=self.embeddings,
            ids=ids)
        return embeddings[0, 0]

    def get_config(self):
        return {}

