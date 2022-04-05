from typing import Text, List

import os
import tensorflow as tf
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata import schema_utils

from tfx import v1 as tfx
from tfx_bsl.public import tfxio

from rexify.models import Recommender

BATCH_SIZE = os.environ.get('BATCH_SIZE', 512)
FEATURE_KEYS = ['userId', 'itemId']
FEATURE_SPEC = {
    feature: tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64)
    for feature in FEATURE_KEYS
}


def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              schema: schema_pb2.Schema,
              batch_size: int = 512) -> tf.data.Dataset:
    # todo: dataset factory not working properly
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size),
        schema).repeat()


def run_fn(fn_args: tfx.components.FnArgs):

    layer_sizes = [64, 32]
    activation = 'leaky_relu'
    schema = schema_utils.schema_from_feature_spec(FEATURE_SPEC)

    training_data: tf.data.Dataset = _input_fn(
        file_pattern=fn_args.train_files,
        data_accessor=fn_args.data_accessor,
        schema=schema,
        batch_size=512)

    nb_users = 10_000  # len(training_data.map(lambda x: x['userId']).apply(tf.data.experimental.unique))
    nb_items = 10_000  # len(training_data.map(lambda x: x['itemId']).apply(tf.data.experimental.unique))

    query_params = {
        'schema': {'userId': 'categorical'},
        'layer_sizes': layer_sizes,
        'activation': activation,
        'params': {
            'userId': {
                'input_dim': nb_items,
                'embedding_dim': 16
            }
        }
    }

    candidate_params = {
        'schema': {'itemId': 'categorical'},
        'layer_sizes': layer_sizes,
        'activation': activation,
        'params': {
            'itemId': {
                'input_dim': nb_users,
                'embedding_dim': 32
            }
        }
    }

    model: Recommender = Recommender(
        query_params=query_params,
        candidate_params=candidate_params,
        layer_sizes=layer_sizes,
        activation=activation)
    model.compile(optimizer=tf.keras.optimizers.Adam(.2))

    model.fit(training_data, steps_per_epoch=fn_args.train_steps)
    model.save(fn_args.serving_model_dir, save_format='tf')
