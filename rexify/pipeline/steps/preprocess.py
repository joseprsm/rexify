from typing import List, Dict, Any

import tensorflow as tf
import tensorflow_transform as tft

from utils import get_header

NUM_OOV_BUCKETS = 1
SCHEMA_TYPES = ['categorical', 'numeric', 'text', 'date']


def _filter_schema_by_type(dtype: str):

    def filter_schema(schema):
        return [k for k, v in schema.items() if v == dtype]

    return filter_schema


def preprocessing_fn(inputs: Dict[str, tf.Tensor],
                     custom_config: Dict[str, Any]):

    outputs = {}
    schema: Dict[str, str] = get_header(schema=custom_config['schema'])

    feature_keys: Dict[str, List[str]] = {
        dtype: _filter_schema_by_type(dtype)(schema)
        for dtype in SCHEMA_TYPES}

    for key in feature_keys['numeric']:
        outputs[key] = tft.scale_to_0_1(inputs[key])

    for key in feature_keys['categorical']:
        outputs[key] = tft.compute_and_apply_vocabulary(
            tf.strings.strip(inputs[key]),
            num_oov_buckets=NUM_OOV_BUCKETS,
            vocab_filename=key)

    # todo: add cyclical transform for date features
    # for key in feature_keys['date']:
    #     for date_feature in ['year', 'month', 'day']:
    #         outputs[f'{key}_{date_feature}'] = ...

    # passing remaining features as they will be transformed during training
    untransformed_features = feature_keys['categorical'] + feature_keys['numeric']
    for key in inputs.keys():
        if key not in untransformed_features and key in schema.keys():
            outputs[key] = inputs[key]

    return outputs
