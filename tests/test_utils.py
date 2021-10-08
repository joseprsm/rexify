import os
import tempfile

import tensorflow as tf
from tfx.types import standard_artifacts

from rexify import utils


def test_get_feature_spec():
    schema = {'itemId': 1}
    feature_spec = utils.get_feature_spec(schema)
    assert isinstance(feature_spec, dict)
    assert isinstance(feature_spec['itemId'], tf.io.FixedLenFeature)
    assert feature_spec['itemId'].shape == list()
    assert feature_spec['itemId'].dtype == tf.int64


def test_load_model():
    model_ = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    model_(tf.constant([[1]]))
    model_name = 'test_model'

    with tempfile.TemporaryDirectory() as tmpdirname:
        model_.save(os.path.join(tmpdirname, model_name))
        model_artifact = standard_artifacts.Model()
        model_artifact.uri = tmpdirname

        model = utils.load_model([model_artifact], model_name)

    assert isinstance(model, tf.keras.Model)
    assert isinstance(model.layers[0], tf.keras.layers.Dense)
