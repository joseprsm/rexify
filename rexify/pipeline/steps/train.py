from typing import Text, List

import tensorflow as tf
import tensorflow_transform as tft
import tfx.components

from tfx_bsl.public import tfxio

from rexify.features.sequence import slide_transform
from rexify.models import Recommender


def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 512) -> tf.data.Dataset:
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size),
        tf_transform_output.transformed_metadata.schema
    ).repeat()


def run_fn(fn_args: tfx.components.FnArgs):

    # todo: read from custom config
    schema = ...
    params = ...
    layer_sizes = ...
    activation = ...

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    training_data: tf.data.Dataset = _input_fn(
        file_pattern=fn_args.train_files,
        data_accessor=fn_args.data_accessor,
        tf_transform_output=tf_transform_output,
        batch_size=512)

    # todo: move sliding window to Transform step?
    training_data = slide_transform(training_data, schema)

    model: tf.keras.Model = Recommender(
        schema=schema,
        params=params,
        layer_sizes=layer_sizes,
        activation=activation)
    model.fit(training_data, steps_per_epoch=fn_args.train_steps)
    model.save(fn_args.serving_model_dir, save_format='tf')
