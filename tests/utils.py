from typing import Dict, Union, Any

import numpy as np
import tensorflow as tf

from rexify.models import Recommender, EmbeddingLookup


def get_header(schema: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    header = dict()
    keys = list(schema.keys())
    for key in keys:
        header.update(schema[key])
    return header


def load_mock_events():

    header = ['userId', 'itemId', 'date']
    events = np.array([
        [1, 2, 1, 1, 2, 2],
        [3, 3, 4, 5, 4, 6],
        [1, 2, 3, 4, 5, 6]]).transpose()

    @tf.autograph.experimental.do_not_convert
    def add_header(x):
        return {header[i]: x[i] for i in range(len(header))}

    return tf.data.Dataset.from_tensor_slices(events).map(add_header)


def get_sample_data():
    candidates = np.arange(1, 29)
    embeddings = np.random.rand(len(candidates), 32).astype(np.float32)
    return candidates, embeddings


def get_sample_schema() -> Dict[str, Union[str, Dict[str, str]]]:
    return {
        'user': {'userId': 'categorical'},
        'item': {'itemId': 'categorical'},
        'date': 'timestamp'}


def get_sample_params() -> Dict[str, Dict[str, Dict[str, Any]]]:
    return {
        'user': {'userId': {'input_dim': 31, 'embedding_dim': 128}},
        'item': {'itemId': {'input_dim': 29, 'embedding_dim': 128}}
    }


def get_tf_datasets(candidates, embeddings):
    candidates_tf = tf.data.Dataset.from_tensor_slices(candidates).map(lambda x: {'itemId': x})
    embeddings_tf = tf.data.Dataset.from_tensor_slices(
        embeddings.reshape((len(candidates), 1, 32))).map(lambda x: tf.cast(x, tf.float32))
    return candidates_tf, embeddings_tf


_candidates, _embeddings = get_sample_data()
_schema = get_sample_schema()
_params = get_sample_params()
mock_candidates, mock_embeddings = get_tf_datasets(_candidates, _embeddings)
mock_recommender = Recommender(schema=_schema, params=_params, layer_sizes=[64, 32])
mock_lookup = EmbeddingLookup(vocabulary=_candidates, embeddings=_embeddings)

