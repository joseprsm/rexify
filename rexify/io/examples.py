from typing import Union, Dict, Any, List

import os
import json
import tensorflow as tf

from tfx.types import Artifact, artifact_utils


def get_feature_spec(schema: Dict[str, str]) -> Dict[str, Any]:
    return {
        key: tf.io.FixedLenFeature([], dtype=tf.int64)  # todo: infer dtype
        for key in schema.keys()
    }


def read(example_uri: Union[str, bytes, os.PathLike],
         feature_spec: Dict[str, Any]) -> tf.data.Dataset:

    def parse_examples(x):
        return tf.io.parse_example(x, features=feature_spec)

    filenames = [os.path.join(example_uri, filename) for filename in tf.io.gfile.listdir(example_uri)]
    examples = tf.data.TFRecordDataset(filenames=filenames, compression_type='GZIP')
    return examples.map(parse_examples)


def read_split_examples(artifact_list: List[Artifact], schema: str, split: str = 'train'):
    feature_spec = get_feature_spec(json.loads(schema))
    examples_uri: Union[str, bytes, os.PathLike] = artifact_utils.get_split_uri(artifact_list, split)
    examples: tf.data.Dataset = read(examples_uri, feature_spec=feature_spec)
    return examples


def export(artifact_list: List[Artifact],
           examples: tf.data.Dataset,
           filename: str):

    serialized: tf.data.Dataset = examples.map(lambda x: {
        key: tf.io.serialize_tensor(x[key])
        for key in examples.element_spec.keys()})

    with tf.io.TFRecordWriter('helo') as writer:
        writer.write(serialized)


def serialize(example: Dict[str, tf.Tensor],
              header: Dict[str, str]):

    def feature_factory(feature_dtype: str, value):
        if feature_dtype == 'categorical' or feature_dtype == 'text':
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))
        elif feature_dtype == 'ordinal':
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        elif feature_dtype == 'continuous':
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        raise NotImplementedError

    features = {
        feature: feature_factory(feature_type, example[feature])
        for feature, feature_type in header.items()}

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()
