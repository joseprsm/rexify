import pandas as pd
import tensorflow as tf

# noinspection PyProtectedMember
from rexify.features.sequence import slide_transform, _filter_by_keys

from tests.utils import get_sample_schema, load_mock_events


def test_filtering():
    schema = get_sample_schema()

    inputs = {
        'userId': tf.constant([1]),
        'itemId': tf.constant([1, 1, 1]),
        'target': tf.constant([1]),
        'date': tf.constant([1])}

    outputs = _filter_by_keys(schema)(inputs)

    def do_assert(key):
        assert outputs[key] == tf.constant([1])
        assert outputs[key].shape == tf.TensorShape([])

    for k in ['userId', 'date', 'itemId']:
        do_assert(k)

    assert tf.reduce_all(outputs['sequence'] == tf.constant([1, 1]))
    assert outputs['sequence'].shape == tf.TensorShape([2])


def test_sliding_window():
    inputs = load_mock_events()
    schema = get_sample_schema()

    outputs = list(slide_transform(inputs, schema, window_size=3))

    assert len(outputs) == 2
    assert outputs[0]['userId'] == tf.constant(1, dtype=tf.int64)
    assert tf.reduce_all(outputs[0]['sequence'] == tf.constant([3, 4], dtype=tf.int64))
    assert outputs[0]['date'] == tf.constant(4, dtype=tf.int64)
    assert outputs[0]['itemId'] == tf.constant(5, dtype=tf.int64)

    assert outputs[1]['userId'] == tf.constant(2, dtype=tf.int64)
    assert tf.reduce_all(outputs[1]['sequence'] == tf.constant([3, 4], dtype=tf.int64))
    assert outputs[1]['date'] == tf.constant(6, dtype=tf.int64)
    assert outputs[1]['itemId'] == tf.constant(6, dtype=tf.int64)
