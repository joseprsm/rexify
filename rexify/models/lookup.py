from typing import Union, Optional, Any

import numpy as np
import tensorflow as tf


class EmbeddingLookup(tf.keras.Model):

    """
    Model that maps IDs to the items' embeddings, allowing a
    more efficient retrieval.

    Args:
        vocabulary: set of identifiers
        embeddings: the vocabulary's respective pretrained embeddings
        sample_query: sample identifier to build and test the model
    """

    def __init__(
        self,
        vocabulary: Union[tf.data.Dataset, np.array],
        embeddings: Union[tf.data.Dataset, np.array],
        sample_query: Optional[Any] = None,
        **kwargs
    ):
        super(EmbeddingLookup, self).__init__(**kwargs)
        self.vocabulary = vocabulary  # sets the attribute by default
        # guarantee the tensor's/array's items are strings
        if isinstance(vocabulary, np.ndarray):
            self.vocabulary = vocabulary.astype(str)
        else:
            if vocabulary.dtype.name != "string":
                self.vocabulary = tf.strings.as_string(vocabulary)
        # sample query is optional to create the model but it's
        # automatically set up during model save, in order to be
        # used in the next pipeline step.
        self.sample_query = sample_query
        self.embeddings = embeddings

        # maps the string identifiers in the vocabulary to integer IDs
        init = tf.lookup.KeyValueTensorInitializer(
            keys=self.vocabulary, values=list(range(len(vocabulary)))
        )
        self.token_to_id = tf.lookup.StaticHashTable(
            init, default_value=len(vocabulary)
        )

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def call(self, inputs, *_):
        tokens = tf.strings.split(inputs, sep=None)
        ids = self.token_to_id.lookup(tokens)
        embeddings = tf.nn.embedding_lookup(params=self.embeddings, ids=ids)
        return embeddings[0, 0]

    def get_config(self):
        # creates a sample_query based on the existent vocabulary
        sample_query = self.vocabulary[0]
        if isinstance(sample_query, tf.Tensor):
            sample_query = sample_query.numpy()
        # if `sample_query` is in bytes format (typically happens when
        # vocabulary if tf.data.Dataset then decode it
        sample_query = (
            sample_query.decode() if isinstance(sample_query, bytes) else sample_query
        )
        # TODO: add serialized vocabulary, embeddings
        return {"sample_query": sample_query}
