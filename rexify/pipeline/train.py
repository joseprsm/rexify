from typing import Text, List, Dict, Any

import os
import tensorflow as tf
import tensorflow_transform as tft
import tfx.components

from tfx_bsl.public import tfxio

from rexify.models import Recommender

BATCH_SIZE = os.environ.get('BATCH_SIZE', 512)


def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 512) -> tf.data.Dataset:
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size),
        tf_transform_output.transformed_metadata.schema).repeat()


def run_fn(fn_args: tfx.components.FnArgs,
           custom_config: Dict[str, Any]):

    layer_sizes = custom_config.get('layer_sizes', [64, 32])
    activation = custom_config.get('activation', 'leaky_relu')

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    training_data: tf.data.Dataset = _input_fn(
        file_pattern=fn_args.train_files,
        data_accessor=fn_args.data_accessor,
        tf_transform_output=tf_transform_output,
        batch_size=512)

    nb_users = len(training_data.map(lambda x: tf.data.experimental.unique(x['userId'])))
    nb_items = len(training_data.map(lambda x: tf.data.experimental.unique(x['itemId'])))

    model: Recommender = Recommender(
        nb_users + 1,
        nb_items + 1,
        layer_sizes=layer_sizes,
        activation=activation)
    model.fit(training_data, steps_per_epoch=fn_args.train_steps)
    model.save(fn_args.serving_model_dir, save_format='tf')
