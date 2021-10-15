import pandas as pd
import tensorflow as tf

# noinspection PyProtectedMember
from rexify.features.sequence import slide_transform, _filter_by_keys


def test_filtering():
    schema = {
        'user': {'userId': 'categorical'},
        'item': {'itemId': 'categorical'},
        'date': 'timestamp'}

    inputs = {
        'userId': tf.constant([1]),
        'itemId': tf.constant([1, 1, 1]),
        'target': tf.constant([1]),
        'date': tf.constant([1])}

    outputs = _filter_by_keys(schema)(inputs)

    def do_assert(key):
        assert outputs[key] == tf.constant([1])
        assert outputs[key].shape == tf.TensorShape([])

    for k in ['userId', 'date', 'target']:
        do_assert(k)

    assert tf.reduce_all(outputs['sequence'] == tf.constant([1, 1]))
    assert outputs['sequence'].shape == tf.TensorShape([2])


def test_sliding_window():
    inputs = pd.DataFrame({
        'userId': [1, 2, 1, 1, 2],
        'itemId': [1, 1, 2, 3, 2],
        'date': [1, 2, 3, 4, 5]})

    inputs = tf.data.Dataset.from_tensor_slices(inputs.values).\
        map(lambda x: {inputs.columns[i]: x[i] for i in range(len(inputs.columns))})

    schema = {'user': {'userId': 'categorical'}, 'item': {'itemId': 'categorical'}, 'date': 'timestamp'}

    outputs = list(slide_transform(inputs, schema, window_size=3))

    assert len(outputs) == 2
    assert outputs[0]['userId'] == tf.constant(1, dtype=tf.int64)
    assert tf.reduce_all(outputs[0]['sequence'] == tf.constant([1, 2], dtype=tf.int64))
    assert outputs[0]['date'] == tf.constant(4, dtype=tf.int64)
    assert outputs[0]['target'] == tf.constant(3, dtype=tf.int64)

    assert outputs[1]['userId'] == tf.constant(2, dtype=tf.int64)
    assert tf.reduce_all(outputs[1]['sequence'] == tf.constant([1], dtype=tf.int64))
    assert outputs[1]['date'] == tf.constant(5, dtype=tf.int64)
    assert outputs[1]['target'] == tf.constant(2, dtype=tf.int64)
