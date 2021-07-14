import string
import numpy as np
import tensorflow as tf

from rexify.models.lookup import EmbeddingLookup


def _get_sample_data():
    vocab = np.random.choice(list(string.ascii_lowercase), 10, replace=False)
    embeddings = np.random.rand(10, 32)
    return vocab, embeddings


def test_lookup():
    vocab, embeddings = _get_sample_data()
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
    vocab, embeddings = _get_sample_data()
    lookup = EmbeddingLookup(
        vocabulary=vocab,
        embeddings=embeddings)
    assert isinstance(lookup.get_config(), dict)


def test_lookup_input_tensors():
    vocab, embeddings = _get_sample_data()
    vocab = tf.constant(vocab)
    embeddings = tf.constant(embeddings)
    sample_query = [vocab[0]]

    lookup = EmbeddingLookup(vocabulary=vocab, embeddings=embeddings)
    try:
        query_embedding = lookup(sample_query)
        assert isinstance(query_embedding, tf.Tensor)
        assert query_embedding.shape[-1] == 32
    except ValueError:
        assert False


# def test_lookup_attributes():
#     vocab, embeddings = _get_sample_data()
#     lookup = EmbeddingLookupModel(vocabulary=vocab, embeddings=embeddings)
#     lookup([lookup.vocabulary[0]])
#
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         model_name = os.path.join(tmpdirname, 'lookup')
#         lookup.save(model_name)
#         new_lookup: EmbeddingLookupModel = tf.keras.models.load_model(model_name)
#
#     assert hasattr(new_lookup, 'vocabulary')
#     assert hasattr(new_lookup, 'embeddings')
