from typing import List, Dict, Any

import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURE_KEYS: List[str] = ...
NUMERIC_FEATURE_KEYS: List[str] = ...
DATE_KEY: str = ...

NUM_OOV_BUCKETS = 1


def preprocessing_fn(inputs: Dict[str, tf.Tensor],
                     custom_config: Dict[str, Any]):

    schema = custom_config['schema']
    outputs = {}

    for key in NUMERIC_FEATURE_KEYS:
        outputs[key] = tft.scale_to_0_1(inputs[key])

    for key in CATEGORICAL_FEATURE_KEYS:
        outputs[key] = tft.compute_and_apply_vocabulary(
            tf.strings.strip(inputs[key]),
            num_oov_buckets=NUM_OOV_BUCKETS,
            vocab_filename=key)

    # todo: add cyclical transform for date features

    # passing remaining features as they will be transformed during training
    # todo: add schema validation
    for key in inputs.keys():
        if key not in CATEGORICAL_FEATURE_KEYS + NUMERIC_FEATURE_KEYS and key in schema.keys():
            outputs[key] = inputs[key]

    return outputs
