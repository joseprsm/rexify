from numbers import Number

import string
import numpy as np
import tensorflow as tf

from rexify.lookup import EmbeddingLookup


def test_lookup():
    vocab = np.random.choice(list(string.ascii_lowercase), 10, replace=False)
    embeddings = np.random.rand(10, 32)
    lookup = EmbeddingLookup(
        vocabulary=vocab,
        embeddings=embeddings)
    assert lookup.token_to_id.key_dtype == tf.string
    value_dtype = lookup.token_to_id.value_dtype
    assert value_dtype == tf.int32 or value_dtype == tf.int64

    emb = lookup([lookup.vocabulary[0]])
    assert emb is not None
    assert isinstance(emb, tf.Tensor)
    assert emb.shape == tf.TensorShape(32)


def test_lookup_config():
    vocab = np.random.choice(list(string.ascii_lowercase), 10, replace=False)
    embeddings = np.random.rand(10, 32)
    lookup = EmbeddingLookup(
        vocabulary=vocab,
        embeddings=embeddings)
    assert lookup.get_config() == dict()
