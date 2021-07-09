import numpy as np
import pandas as pd
import tensorflow as tf

from rexify.recommender import Recommender


def _load_mock_data():
    events = pd.DataFrame({
        'userId': np.random.choice(
            list(range(1, 31)), 1024),
        'itemId': np.random.choice(
            list(range(67, 95)), 1024)})

    def add_header(x):
        return {events.columns[i]: x[i] for i in range(len(events.columns))}

    return tf.data.Dataset. \
        from_tensor_slices(events.values). \
        map(add_header)


def test_recommender_training():
    events = _load_mock_data()

    rex = Recommender(50, 50)
    rex.compile(optimizer=tf.keras.optimizers.Adagrad(0.2))
    assert rex.history is None

    rex.fit(events.batch(512))
    assert rex.history is not None


def test_recommender_call():
    rex = Recommender(50, 50)
    sample_query = {'userId': tf.constant([1]), 'itemId': tf.constant([1])}
    query_embeddings, candidate_embeddings = rex(sample_query)
    assert isinstance(query_embeddings, tf.Tensor)
    assert isinstance(candidate_embeddings, tf.Tensor)

    assert query_embeddings.shape == tf.TensorShape([1, 32])
    assert candidate_embeddings.shape == tf.TensorShape([1, 32])

    assert (tf.reduce_sum(
        tf.cast(
            query_embeddings == rex.query_model(tf.constant([[1]])),
            tf.int32)) == 32).numpy()


def test_recommender_get_config():
    rex = Recommender(50, 50)
    assert isinstance(rex.get_config(), dict)
    assert rex.get_config() == dict()
