from typing import Union, Optional, Any

import numpy as np
import tensorflow as tf


class EmbeddingLookup(tf.keras.Model):

    def __init__(
            self,
            vocabulary: Union[tf.data.Dataset, np.array],
            embeddings: Union[tf.data.Dataset, np.array],
            sample_query: Optional[Any] = None,
            **kwargs
    ):
        super(EmbeddingLookup, self).__init__(**kwargs)
        self.vocabulary = tf.strings.as_string(vocabulary) if 'str' not in vocabulary.dtype.name else vocabulary
        self.sample_query = sample_query
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
        sample_query = self.vocabulary[0]
        if isinstance(sample_query, tf.Tensor):
            sample_query = sample_query.numpy().decode()
        return {'sample_query': sample_query}
