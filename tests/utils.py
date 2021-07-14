import string
import numpy as np
import pandas as pd
import tensorflow as tf

from rexify.models import Recommender, EmbeddingLookup


def load_mock_events():
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


def get_sample_data():
    candidates = np.array(list(set([
        ''.join(x) for x in np.random.choice(
            list(string.ascii_lowercase), (100, 3))])))
    embeddings = np.random.rand(len(candidates), 32).astype(np.float32)
    return candidates, embeddings


def get_tf_datasets(candidates, embeddings):
    candidates_tf = tf.data.Dataset.from_tensor_slices(candidates).map(lambda x: {'itemId': x})
    embeddings_tf = tf.data.Dataset.from_tensor_slices(
        embeddings.reshape((len(candidates), 1, 32))).map(lambda x: tf.cast(x, tf.float32))
    return candidates_tf, embeddings_tf


_candidates, _embeddings = get_sample_data()
mock_candidates, mock_embeddings = get_tf_datasets(_candidates, _embeddings)
mock_recommender = Recommender(100, 100)
mock_lookup = EmbeddingLookup(vocabulary=_candidates, embeddings=_embeddings)

