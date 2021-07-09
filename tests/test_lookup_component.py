import numpy as np
import tensorflow as tf

from rexify.models.recommender import Recommender
from rexify.pipeline.components.lookup.executor import Executor

from utils import load_mock_events


def test_get_lookup_params():
    feature_key = 'userId'
    examples: tf.data.Dataset = load_mock_events().map(
        lambda x: x[feature_key]).apply(
        tf.data.experimental.unique()).map(
        lambda x: {feature_key: x})

    vocabulary, embeddings = Executor.get_lookup_params(
        examples=examples,
        model=Recommender(50, 50),
        query_model='candidate_model',
        feature_key=feature_key)

    assert isinstance(vocabulary, np.ndarray)
    assert isinstance(embeddings, np.ndarray)
    assert vocabulary.shape[0] == 30
    assert embeddings.shape[0] == 30 and embeddings.shape[-1] == 32
