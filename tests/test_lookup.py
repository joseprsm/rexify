import string
import numpy as np
import tensorflow as tf

from rexify.models.lookup import EmbeddingLookupModel


def test_lookup():
    vocab = np.random.choice(list(string.ascii_lowercase), 10, replace=False)
    embeddings = np.random.rand(10, 32)
    lookup = EmbeddingLookupModel(
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
    lookup = EmbeddingLookupModel(
        vocabulary=vocab,
        embeddings=embeddings)
    assert lookup.get_config() == dict()


def test_lookup_input_tensors():
    vocab = np.random.choice(list(string.ascii_lowercase), 10, replace=False)
    sample_query = [vocab[0]]
    vocab = tf.constant(vocab)
    embeddings = tf.constant(np.random.rand(10, 32))
    lookup = EmbeddingLookupModel(vocabulary=vocab, embeddings=embeddings)
    try:
        query_embedding = lookup(sample_query)
        assert isinstance(query_embedding, tf.Tensor)
        assert query_embedding.shape[-1] == 32
    except ValueError:
        assert False
