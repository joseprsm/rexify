from typing import Dict, Any

import tensorflow as tf
import tensorflow_transform as tft

NUM_OOV_BUCKETS = 1


def preprocessing_fn(inputs: Dict[str, tf.Tensor],
                     custom_config: Dict[str, Any]):

    schema = custom_config['schema']
    flat_schema = schema['user']
    flat_schema.update(schema['item'])
    flat_schema.update(schema['context'])

    feature_keys = {
        dtype: [key for key, value in flat_schema.items() if value == dtype]
        for dtype in set(flat_schema.values())
    }

    categorical_feature_keys = feature_keys.pop('categorical')
    numeric_feature_keys = feature_keys.pop('numeric')
    # date_feature_key = feature_keys.pop('timestamp')

    outputs = {}

    for key in numeric_feature_keys:
        outputs[key] = tft.scale_to_0_1(inputs[key])

    for key in categorical_feature_keys:
        outputs[key] = tft.compute_and_apply_vocabulary(
            tf.strings.strip(inputs[key]),
            num_oov_buckets=NUM_OOV_BUCKETS,
            vocab_filename=key)

    # todo: add cyclical transform for date features

    # passing remaining features as they will be transformed during training
    # todo: add schema validation
    for key in inputs.keys():
        if key not in categorical_feature_keys + numeric_feature_keys and key in flat_schema.keys():
            outputs[key] = inputs[key]

    return outputs
