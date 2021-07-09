import string
import numpy as np
import tensorflow as tf

from rexify.models.scann import ScaNN
from rexify.models.lookup import EmbeddingLookupModel


def get_sample_data():
    candidates = np.array(list(set([
        ''.join(x) for x in np.random.choice(
            list(string.ascii_lowercase), (1_000, 2))])))
    embeddings = np.random.rand(len(candidates), 32).astype(np.float32)
    return candidates, embeddings


def get_tf_datasets(candidates, embeddings):
    candidates_tf = tf.data.Dataset.from_tensor_slices(candidates)
    embeddings_tf = tf.data.Dataset.from_tensor_slices(
        embeddings.reshape((len(candidates), 1, 32))).map(lambda x: tf.cast(x, tf.float32))
    return candidates_tf, embeddings_tf


def get_sample_positive_inputs():
    candidates, embeddings = get_sample_data()
    lookup_model = EmbeddingLookupModel(vocabulary=candidates, embeddings=embeddings)

    sample_query = candidates[0]
    candidates_tf, embeddings_tf = get_tf_datasets(candidates, embeddings)
    return lookup_model, candidates_tf, embeddings_tf, sample_query


def test_scann():
    inputs = get_sample_positive_inputs()
    sample_query = inputs[-1]
    scann = ScaNN(*inputs, k=10)
    similarity, ids = scann([sample_query])
    assert isinstance(similarity, tf.Tensor)
    assert similarity.shape == tf.TensorShape([10])

    assert isinstance(ids, tf.Tensor)
    assert ids.shape == tf.TensorShape([10])


def test_scann_config():
    scann = ScaNN(*get_sample_positive_inputs(), k=10)
    conf = scann.get_config()
    assert isinstance(conf, dict)
    assert list(conf.keys()) == ['lookup_model', 'candidates', 'embeddings']
