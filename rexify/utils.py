from typing import Text, Dict, Any

import os
import tensorflow as tf


def read_split_examples(
        example_uri: Dict[Text, Text],
        feature_spec: Dict[Text, Any]) -> Dict[Text, tf.data.Dataset]:

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
