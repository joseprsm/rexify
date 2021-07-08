import pandas as pd
import tensorflow as tf

from rexify.recommender import Recommender


def _load_mock_data():
    events = pd.read_csv('data/events.csv').head(1024)

    def add_header(x):
        return {events.columns[i]: x[i] for i in range(len(events.columns))}

    return tf.data.Dataset. \
        from_tensor_slices(events.values). \
        map(add_header)


def test_model_training():
    events = _load_mock_data()

    rex = Recommender(250_000, 10_000)
    rex.compile(optimizer=tf.keras.optimizers.Adagrad(0.2))

    rex.fit(events.batch(512))
    assert rex.history is not None
