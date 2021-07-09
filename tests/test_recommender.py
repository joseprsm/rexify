import tensorflow as tf

from rexify.models.recommender import Recommender

from utils import load_mock_events


def test_recommender_training():
    events = load_mock_events()

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

    query_embedding_1 = rex.query_model(tf.constant([[1]]))
    assert (tf.reduce_sum(
        tf.cast(
            query_embeddings == query_embedding_1,
            tf.int32)
    ) == 32).numpy()


def test_recommender_get_config():
    rex = Recommender(50, 50)
    assert isinstance(rex.get_config(), dict)
    assert rex.get_config() == dict()
