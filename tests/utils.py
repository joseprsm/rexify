import numpy as np
import pandas as pd
import tensorflow as tf


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
