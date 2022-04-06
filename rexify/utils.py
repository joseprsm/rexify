from typing import Text, Dict, Any, List, Union
from ast import literal_eval

import os
import tensorflow as tf
from tfx.dsl.io import fileio

from tfx.types import Artifact, artifact_utils


def read_examples(
        example_uri: Union[str, bytes, os.PathLike],
        feature_spec: Dict[Text, Any]) -> tf.data.Dataset:

    def parse_examples(x):
        return tf.io.parse_example(x, features=feature_spec)

    filenames = [os.path.join(example_uri, filename) for filename in tf.io.gfile.listdir(example_uri)]
    examples = tf.data.TFRecordDataset(filenames=filenames, compression_type='GZIP')
    return examples.map(parse_examples)


def get_feature_spec(schema: Dict[Text, Text]) -> Dict[Text, Any]:
    return {
        key: tf.io.FixedLenFeature([], dtype=tf.int64)
        for key in schema.keys()
    }


def load_model(artifact_list: List[Artifact], model_name: Text = 'Format-Serving'):
    model_uri: Text = artifact_utils.get_single_uri(artifact_list)
    model: tf.keras.Model = tf.keras.models.load_model(os.path.join(model_uri, model_name))
    return model


def read_split_examples(artifact_list: List[Artifact], schema, split: Text = 'train'):
    feature_spec = get_feature_spec(literal_eval(schema))
    examples_uri: Text = artifact_utils.get_split_uri(artifact_list, split)
    examples: tf.data.Dataset = read_examples(examples_uri, feature_spec=feature_spec)
    return examples


def export_model(artifact_list: List[Artifact], model: tf.keras.Model, model_name: Text, **kwargs):
    output_dir = artifact_utils.get_single_uri(artifact_list)
    output_uri = os.path.join(output_dir, model_name)
    fileio.makedirs(os.path.dirname(output_uri))
    model.save(output_uri, save_format='tf', **kwargs)
