from typing import Text, List

import tensorflow as tf
import tensorflow_recommenders as tfrs

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata import schema_utils
from tfx.components.trainer.fn_args_utils import DataAccessor, FnArgs
from tfx_bsl.public import tfxio

from rexify.models.recommender import Recommender

_FEATURE_KEYS = ['userId', 'itemId']
_FEATURE_SPEC = {
    feature: tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
    for feature in _FEATURE_KEYS
}


def _input_fn(file_pattern: List[Text],
              data_accessor: DataAccessor,
              schema: schema_pb2.Schema,
              batch_size: int = 512) -> tf.data.Dataset:
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size),
        schema).repeat()


def _build_model() -> tfrs.Model:
    model = Recommender(250_000, 50_000)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.2))
    return model


def run_fn(fn_args: FnArgs):
    schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        schema,
        batch_size=512)

    model = _build_model()
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps)
    model.save(fn_args.serving_model_dir, save_format='tf')
